import numpy as np

import copy

import os

if os.name == 'nt':
    from kullback_leibler import klucbBern, klucbGauss
else:
    import pyximport
    _ = pyximport.install()
    from kullback_leibler_cython import klucbBern, klucbGauss


class PlayerStrategy():
    def __init__(self, narms, T):
        self.T = T  # horizon
        self.t = 0  # current round
        self.K = narms  # number of arms
        self.means = np.zeros(narms)  # empirical means
        self.B = np.inf * np.ones(narms)  # confidence bound
        self.npulls = np.zeros(narms)  # number of pulls for each arm

    def set_graph_mode(self, relative_id, interface):
        raise Exception('Not implemented!')


class SingleAgentKLUCB(PlayerStrategy):
    def __init__(self, narms, T):
        PlayerStrategy.__init__(self, narms, T)
        self.state_round = 0
        self.arms_reward = [0 for i in range(self.K)]
        self.arms_nselected = [0 for i in range(self.K)]
        self.arms_kl_index = [np.inf for i in range(self.K)]
        self.arms_kl_index[0] = 0
        self.arms_emp_avg_reward = [0 for i in range(self.K)]
        self.arms_to_explore = [i + 1 for i in range(self.K)]
        self.arms_to_explore.pop(0)
        self.is_leader = True

    def play(self):
        return np.argmax(self.arms_kl_index)

    def _kl_index_update(self):
        # KL Indexes update
        if self.state_round > 2:
            f_t = np.log(
                self.state_round) + 4 * np.log(np.log(self.state_round))
            for k in range(self.K):
                if self.arms_nselected[k] > 0:
                    f_t_k = f_t / self.arms_nselected[k]
                    self.arms_kl_index[k] = klucbBern(
                        self.arms_emp_avg_reward[k], f_t_k)
                else:
                    self.arms_kl_index[k] = np.inf

    def update(self, play, obs):
        self.state_round += 1
        arm = play + 1
        rew, col = obs
        self.arms_nselected[arm - 1] += 1
        self.arms_reward[arm - 1] += rew
        self.arms_emp_avg_reward[
            arm - 1] = self.arms_reward[arm - 1] / self.arms_nselected[arm - 1]
        self._kl_index_update()


class GraphDPE(PlayerStrategy):
    def __init__(self, narms, T):
        PlayerStrategy.__init__(self, narms, T)
        self.relative_position = None
        self.is_leader = None
        self.communication_interface = None
        self.best_arm = 1
        self.state_round = 0
        self.arms_reward = [0 for i in range(self.K)]
        self.arms_nselected = [0 for i in range(self.K)]
        self.arms_kl_index = [np.inf for i in range(self.K)]
        self.arms_kl_index[0] = 0
        self.arms_emp_avg_reward = [0 for i in range(self.K)]
        self.arms_to_explore = [i + 1 for i in range(self.K)]
        self.arms_to_explore.pop(0)
        self.s = True
        self.last_timestamp = -1
        self.n_times_change_of_arm = []

    def sanity_check(self):
        if self.relative_position is None or \
             self.is_leader is None or self.communication_interface is None:
            return False
        return True

    def set_graph_mode(self, relative_id, interface):
        self.relative_position = relative_id
        self.communication_interface = interface
        self.is_leader = True if self.relative_position == 0 else False

    def _B_update(self):
        #  Update set of arms to explore
        self.arms_to_explore = []
        for arm in range(self.K):
            if self.arms_kl_index[arm] > self.arms_emp_avg_reward[self.best_arm
                                                                  - 1]:
                self.arms_to_explore.append(arm + 1)
        self.s = len(self.arms_to_explore) > 0

    def _kl_index_update(self):
        # KL Indexes update,
        if self.state_round > 2:
            f_t = np.log(
                self.state_round) + 4 * np.log(np.log(self.state_round))
            for k in range(self.K):
                if self.arms_nselected[k] > 0:
                    f_t_k = f_t / self.arms_nselected[k]
                    self.arms_kl_index[k] = klucbBern(
                        self.arms_emp_avg_reward[k], f_t_k)
                else:
                    self.arms_kl_index[k] = np.inf

    def _update_arm_stats(self, arm, rew, col):
        self.arms_nselected[arm - 1] += 1
        self.arms_reward[arm - 1] += rew
        self.arms_emp_avg_reward[
            arm - 1] = self.arms_reward[arm - 1] / self.arms_nselected[arm - 1]

    def play(self):
        if not self.sanity_check():
            raise Exception('Sanity check not passed!')

        if not self.is_leader:
            a = self.best_arm
        else:
            if len(self.arms_to_explore) == 0:
                a = self.best_arm
            else:
                if self.s:
                    a = self.best_arm
                    self.s = False
                else:
                    kl_idxs = [
                        self.arms_kl_index[i - 1] for i in self.arms_to_explore
                    ]
                    b = np.argmax(kl_idxs)
                    a = self.arms_to_explore[b]
                    self.arms_to_explore.pop(b)
        return a - 1

    def update(self, play, obs):
        if not self.sanity_check():
            raise Exception('Sanity check not passed!')
        self.state_round += 1
        arm = play + 1
        rew, col = obs
        self._update_arm_stats(arm, rew, col)

        if self.is_leader:
            if len(self.arms_to_explore) == 0:
                self._kl_index_update()
                self._B_update()
                temp_best_arm = np.argmax(self.arms_emp_avg_reward) + 1
                if temp_best_arm != self.best_arm:
                    self.best_arm = temp_best_arm
                    self.last_timestamp += 1
                    self.n_times_change_of_arm.append(self.state_round)
                    self.communication_interface.send(self.last_timestamp,
                                                      self.best_arm)
        else:
            if not self.communication_interface.is_empty():
                msgs = self.communication_interface.receive()
                new_arm = None
                for t, m in msgs:
                    if m != self.best_arm and t > self.last_timestamp:
                        new_arm = m
                        self.last_timestamp = t
                if new_arm:
                    self.best_arm = new_arm
            self.communication_interface.send(self.last_timestamp,
                                              self.best_arm)


class DPE(PlayerStrategy):
    """
    Decentralized Parsimonious Exploration
    """
    INIT = 0
    INIT_ORTHOG_SAMPLE = 1
    INIT_ORTHOG_VERIFICATION = 2
    INIT_RANK_ASSIGN = 3
    EXPLOIT = 10
    COMMUNICATION = 11
    START_OF_COMMUNICATION = 99  # used just for logging

    phase_names = {
        INIT: 'INIT',
        INIT_ORTHOG_SAMPLE: 'INIT_ORTHOG_SAMPLE',
        INIT_ORTHOG_VERIFICATION: 'INIT_ORTHOG_VERIFICATION',
        INIT_RANK_ASSIGN: 'INIT_RANK_ASSIGN',
        EXPLOIT: 'EXPLOIT',
        COMMUNICATION: 'COMMUNICATION',
        START_OF_COMMUNICATION: 'START_OF_COMMUNICATION'
    }

    def __init__(self,
                 narms,
                 centralized=None,
                 T=10,
                 verbose=0,
                 disable_comp=False,
                 dpe2=False,
                 dpe2_param=None):
        PlayerStrategy.__init__(self, narms, T)
        self.phase = self.INIT_ORTHOG_SAMPLE
        self.id = 0
        self.num_players = 0
        self.backlog_check = False
        self.relative_position = 0
        self.is_leader = False
        self.state_round = 1
        self.collisions = 0  # Used to count collisions
        self.log = []  # Logging feature
        self.log_counter = 0  # Log id
        self.i = 0  # Used during rank assignment
        self.best_arms_set = []
        self.best_arms_min_emp_reward = (1, 0)  # (id, avg emp reward)
        self.arms_to_explore = []  # Used by the leader
        self.cdpe = centralized
        self.disable_comp = disable_comp  # used to disable computations
        self.verbose = verbose
        self.total_time = 0
        self.dpe2 = dpe2  # Block version of DPE

        if self.verbose == 1:
            self.log_func = lambda x: print(x)
        else:
            self.log_func = lambda x: self.log.append(x)

        self.changes_in_best_arms_set = []

        self.arms_reward = [0 for i in range(self.K)]
        self.arms_nselected = [0 for i in range(self.K)]
        self.arms_kl_index = [0 for i in range(self.K)]
        self.arms_emp_avg_reward = [0 for i in range(self.K)]

        self.arms_to_remove = []
        self.arms_to_add = []
        if self.cdpe:
            self.players = None
            self.leader = None
            self.backlog = [[0 for i in range(self.K)],
                            [0 for i in range(self.K)]]

        # Variables used during communication
        self.t0 = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.total_collisions = 0

        # These stats are updated every time we play an arm,
        # therefore they have to be divided by the number of players
        self.stats = {
            self.INIT_ORTHOG_SAMPLE: 0,
            self.INIT_ORTHOG_VERIFICATION: 0,
            self.INIT_RANK_ASSIGN: 0,
            self.EXPLOIT: 0,
            self.COMMUNICATION: 0,
            self.START_OF_COMMUNICATION: 0
        }

        if self.dpe2:
            self._add_log('DPE2 Enabled!')
            if not dpe2_param:
                self.J = int(np.ceil(self.K**0.5))
            else:
                self.J = int(dpe2_param)
            print('J:{}'.format(self.J))

    def _log_phase(self):
        self._add_log('Phase is {}'.format(self.phase_names[self.phase]))

    def _log_id(self):
        self._add_log('Player id: {}'.format(self.id))

    def _add_log(self, text):
        if self.verbose > 0:
            self.log_func('[{}] - {}'.format(self.state_round, text))
            self.log_counter += 1

    def play(self):
        """
        """
        self._log_phase()
        self._log_id()
        self.stats[self.phase] += 1
        # In the init phase we do sampling and verification
        # to assign uniquely the arms
        if self.phase >= self.INIT and self.phase < self.EXPLOIT:
            # Used to identify the round
            self._add_log('State round: {}'.format(self.state_round))

            # In sampling we choose an arm uniformly if the ID is 0, otherwise
            # we choose id.
            if self.phase == self.INIT_ORTHOG_SAMPLE:
                # We don't sample the k-th arm, reserved for verification
                a = self.id if self.id > 0 else np.random.choice(
                    [i + 1 for i in range(self.K - 1)])

            # In verification, depending on the round and our ID, we choose our
            # ID or arm K.
            elif self.phase == self.INIT_ORTHOG_VERIFICATION:
                if self.state_round != self.id and self.id > 0:
                    a = self.id
                else:
                    a = self.K

            # The goal of this phase is to understand how many Players
            # there are, our relative position, and if we are the leader.
            elif self.phase == self.INIT_RANK_ASSIGN:
                a = self.id + max(0, self.i) if (
                    self.id + self.i <= self.K) else self.id
                self.i += 1 if (self.state_round >= 2 * self.id - 1) else 0
        elif self.phase == self.EXPLOIT:
            if not self.is_leader:
                a = self.best_arms_set[(
                    (self.state_round + self.relative_position) %
                    self.num_players)]
            else:
                m = ((self.state_round + self.relative_position) %
                     self.num_players)
                exploration_set = self.arms_to_explore
                if self.dpe2:
                    j = int(
                        np.floor(((self.state_round - m) / self.num_players %
                                  self.J)))
                    exploration_set = self.arms_to_explore[j]
                if len(exploration_set) == 0 or self.best_arms_set[
                        m] != self.best_arms_min_emp_reward[0]:
                    a = self.best_arms_set[m]
                else:
                    if np.random.choice([0, 1]) == 0:
                        # Play arm with lowest empirical mean
                        a = self.best_arms_min_emp_reward[0]
                    else:
                        # Choose arm uniformly from B

                        a = np.random.choice(exploration_set)
        elif self.phase == self.COMMUNICATION:
            if self.state_round > self.t0 and self.state_round <= self.t1 and self.is_leader:
                a = self.best_arms_set[((self.t0 + 1) % self.num_players)]
            elif self.state_round > self.t1 and self.state_round <= self.t2 and self.is_leader:
                a = self.arms_to_remove[0]
                self._add_log(
                    'Leader broadcasting arm to remove: {}'.format(a))
            elif self.state_round > self.t2 and self.state_round <= self.t3 and self.is_leader:
                a = self.arms_to_add[0]
                self._add_log('Leader broadcasting arm to add: {}'.format(a))
            elif not self.is_leader:
                if self.state_round > self.t2:
                    a = ((self.state_round + self.relative_position) %
                         self.K) + 1
                else:
                    a = self.best_arms_set[(
                        (self.state_round + self.relative_position) %
                        self.num_players)]
            else:
                raise Exception('Someone ended up in a wrong state!')

        else:
            self._add_log('Unknown phase: {}'.format(self.phase))
            raise Exception('Unknown phase: {}'.format(self.phase))
        self._add_log('Chose arm: {}'.format(a))
        return a - 1

    def _update_arm_stats(self, arm, rew, col):
        self._add_log('Played arm {} - Reward: {} - Collision: {}'.format(
            arm, rew, col))
        if col == 0:
            # We care about reward only before starting exploiting/exploring or if we
            # are the leader
            self.arms_nselected[arm - 1] += 1
            self.arms_reward[arm - 1] += rew
            self.arms_emp_avg_reward[
                arm -
                1] = self.arms_reward[arm - 1] / self.arms_nselected[arm - 1]
            if self.cdpe and not self.is_leader:
                if not self.leader or not self.players:
                    self.players, self.leader = self.cdpe.centralized()
                # If the leader is not present, just update all the other players also
                if not self.leader:
                    self.backlog[0][arm - 1] += 1
                    self.backlog[1][arm - 1] += rew
                else:
                    if not self.backlog_check:
                        self.backlog_check = True
                        for i in range(self.K):
                            self.leader.arms_nselected[i] += self.backlog[0][i]
                            self.leader.arms_reward[i] += self.backlog[1][i]
                            if self.leader.arms_nselected[i] > 0:
                                self.leader.arms_emp_avg_reward[
                                    i] = self.leader.arms_reward[
                                        i] / self.leader.arms_nselected[i]
                    self.leader.arms_nselected[arm - 1] += 1
                    self.leader.arms_reward[arm - 1] += rew
                    self.leader.arms_emp_avg_reward[
                        arm - 1] = self.leader.arms_reward[
                            arm - 1] / self.leader.arms_nselected[arm - 1]

    def _kl_index_update(self):
        # KL Indexes update
        if not self.disable_comp:
            f_t = np.log(
                self.state_round) + 4 * np.log(np.log(self.state_round))
            self._add_log(
                'Starting update of KL-UCB idxs, f(t): {}'.format(f_t))
            for k in range(self.K):
                if self.arms_nselected[k] > 0:
                    f_t_k = f_t / self.arms_nselected[k]
                    self.arms_kl_index[k] = klucbBern(
                        self.arms_emp_avg_reward[k], f_t_k)
                else:
                    self.arms_kl_index[k] = np.inf
                self._add_log('Updating KL-UCB index for arm {}. b: {}'.format(
                    k, self.arms_kl_index[k]))

    def _B_update(self, arms):
        #  Update set of arms to explore
        if not self.disable_comp:
            if not self.dpe2:
                self.arms_to_explore = []
                for ra in arms:
                    if self.arms_kl_index[
                            ra - 1] >= self.best_arms_min_emp_reward[1]:
                        self.arms_to_explore.append(ra)
            else:
                self.arms_to_explore = [[] for i in range(self.J)]
                for s in range(self.J):
                    for ra in arms:
                        if (s - 1) * self.J <= ra and ra < self.J * s:
                            if self.arms_kl_index[
                                    ra -
                                    1] >= self.best_arms_min_emp_reward[1]:
                                self.arms_to_explore[s].append(ra)

    def _update_best_arms_min_emp_rew(self):
        # Used to obtain mu_Mhat
        if not self.disable_comp:
            self.best_arms_min_emp_reward = [1, np.inf]
            for arm in self.best_arms_set:
                if self.arms_emp_avg_reward[
                        arm - 1] < self.best_arms_min_emp_reward[1]:
                    self.best_arms_min_emp_reward[0] = arm
                    self.best_arms_min_emp_reward[
                        1] = self.arms_emp_avg_reward[arm - 1]

    def _best_arms_update(self):
        if not self.disable_comp:
            M_minus = set(self.best_arms_set)

            # Get a list of the sorted arms according to the empirical avg reward
            sorted_arms = np.argsort(self.arms_emp_avg_reward)
            # The new set of best arms has the first M elements of the list of sorted arms
            new_best_arms = [i + 1 for i in sorted_arms[-self.num_players:]]

            M_plus = set(new_best_arms)

            # Compute arms to be removed/added
            self.arms_to_remove = list(M_minus - M_plus)
            self.arms_to_add = list(M_plus - M_minus)

            L = len(self.arms_to_remove)

            if L != len(self.arms_to_add):
                self._add_log(
                    'Arms to remove and arms to add have different size!'
                    'To remove: {} - To add: {}'.format(
                        self.arms_to_remove, self.arms_to_add))
                raise Exception(
                    'Arms to remove and arms to add have different size!'
                    'To remove: {} - To add: {}'.format(
                        self.arms_to_remove, self.arms_to_add))

            if L > 0:
                self.changes_in_best_arms_set.append((self.total_time, L))
                self._add_log(
                    'Leader: arms to remove {} - arms to add {}'.format(
                        self.arms_to_remove, self.arms_to_add))

                if self.cdpe:
                    self._add_log(
                        'Leader is communicating immediately the change of arm (CENTRALIZED CASE).'
                    )
                    self._centralized_update()
            # Update set B
            self._B_update([i + 1 for i in sorted_arms[:-self.num_players]])

            return True if (L > 0 and not self.cdpe) else False

    def _centralized_update(self):
        if not self.disable_comp:
            if not self.cdpe:
                raise Exception(
                    'Tried to do a centralized update, but CDPE is not enabled!.'
                )
            if self.is_leader:
                while len(self.arms_to_remove) > 0:
                    ar = self.best_arms_set.index(self.arms_to_remove.pop(0))
                    self.best_arms_set[ar] = self.arms_to_add.pop(0)
                self.best_arms_set = np.sort(self.best_arms_set).tolist()
                if not self.players:
                    self.players, _ = self.cdpe.centralized()
                for p in self.players:
                    p.best_arms_set = self.best_arms_set
                self._add_log('Centralized update finished')
            else:
                self._add_log('A follower tried to do a centralized update!')
                raise Exception('A follower tried to do a centralized update!')

    def update(self, play, obs):
        self.total_time += 1
        arm = play + 1
        rew, col = obs
        self.total_collisions += col
        if col > 0 and self.phase >= self.EXPLOIT:
            self._add_log('Had a collision! Playing {}'.format(arm))
        self._update_arm_stats(arm, rew, col)
        # If we are in Sampling phase, we move to verification and
        # assign state 'arm' if no collision happened
        if self.phase == self.INIT_ORTHOG_SAMPLE:
            self.phase = self.INIT_ORTHOG_VERIFICATION
            self.collisions = 0
            self.state_round = 1
            if self.id == 0:
                self.id = arm if col == 0 else 0
            self._add_log('ID is: {}'.format(self.id))

        # In verification phase we count the number of collisions we make
        # If at the end we have no collisions we move to the next phase,
        # otherwise we restart the process from the sampling phase
        elif self.phase == self.INIT_ORTHOG_VERIFICATION:
            self.collisions += col
            if self.state_round == self.K:
                if self.collisions == 0:
                    self.phase = self.INIT_RANK_ASSIGN
                    self.state_round = 1
                    self.collisions = 0
                else:
                    self.phase = self.INIT_ORTHOG_SAMPLE
                    self.state_round = 1
                    self.collisions = 0
            else:
                self.state_round += 1

        elif self.phase == self.INIT_RANK_ASSIGN:
            # Increase number of collisions, this will be equal to M at the end of
            # the round.
            self.collisions += col
            if self.state_round == 2 * self.id - 1:
                self.relative_position = self.collisions + 1
                self._add_log('Identified relative position: {}'.format(
                    self.relative_position))
                if self.collisions == 0:
                    self.is_leader = True
                    self._add_log('We are leader')

            self.state_round += 1
            # Finished all the rounds, move to next phase
            if self.state_round == 2 * self.K - 2:
                # At the end of our block we check how many collisions we
                # had. That number is the number of players
                self.num_players = self.collisions + 1
                self.collisions = 0
                self._add_log('Identified number of players: {}'.format(
                    self.num_players))
                self.best_arms_set = [i + 1 for i in range(self.num_players)]
                # If we are the leader, we need to update the kl-indexes in order to understand
                # which arms to explore
                if self.is_leader:
                    self._kl_index_update()
                    self._B_update([
                        self.num_players + 1 + i
                        for i in range(self.K - self.num_players)
                    ])
                self.phase = self.EXPLOIT

        elif self.phase == self.EXPLOIT:
            self.state_round += 1
            if self.is_leader:
                # Check if we have to do update of the best arms
                m = self.state_round % self.num_players
                if self.dpe2:
                    j = ((self.state_round - m) / self.num_players % self.J)
                    m = m == j
                else:
                    m = m == 0
                if m:
                    self._kl_index_update()
                    if self._best_arms_update():
                        self.phase = self.COMMUNICATION
                        self.stats[self.START_OF_COMMUNICATION] += 1
                        self.t0 = self.state_round - 1
                        self.t1 = self.t0 + self.num_players - 1
                        self.t2 = self.t1 + self.num_players
                        self.t3 = self.t2 + self.K
                        self._add_log(
                            'Leader started communication: t0 {} t1 {} t2 {}'.
                            format(self.t0, self.t1, self.t2))
            elif not self.is_leader and col > 0:
                # Leader is starting communication
                self.t0 = self.state_round - (
                    2 + self.num_players - self.relative_position)
                self.t1 = self.t0 + self.num_players - 1
                self.t2 = self.t1 + self.num_players
                self.t3 = self.t2 + self.K
                self._add_log(
                    'Leader started communication: t0 {} t1 {} t2 {}'.format(
                        self.t0, self.t1, self.t2))
                self.phase = self.COMMUNICATION
            elif not self.is_leader and self.cdpe and col > 0:
                self._add_log(
                    'There has been a collision, even though this is a follower and we are in the centralized case!'
                )
                raise Exception(
                    'There has been a collision, even though this is a follower and we are in the centralized case!'
                )

        elif self.phase == self.COMMUNICATION:
            if self.cdpe:
                self._add_log(
                    'We switched to communication phase though this is the centralized case! Mrel: {}'
                    .format(self.relative_position))
                raise Exception(
                    'We switched to communication phase though this is the centralized case! Mrel: {}'
                    .format(self.relative_position))
            self.state_round += 1
            if not self.is_leader and col > 0:
                if self.state_round - 1 >= self.t1 and self.state_round - 1 <= self.t2:
                    self._add_log('Arm to remove: {}'.format(
                        self.is_leader, arm))
                    self.arms_to_remove = [arm]
                    if arm not in self.best_arms_set:
                        raise Exception('Arm to be removed not present!')
                elif self.state_round - 1 > self.t2 and self.state_round - 1 <= self.t3:
                    self._add_log('Arm to add:{}'.format(self.is_leader, arm))
                    self.arms_to_add = [arm]

            if self.state_round == self.t3 + 1:
                if len(self.arms_to_remove) > 0:
                    ar = self.best_arms_set.index(self.arms_to_remove.pop(0))
                    self.best_arms_set[ar] = self.arms_to_add.pop(0)
                    self.best_arms_set = np.sort(self.best_arms_set).tolist()
                self._add_log('Ended communication, new set of arms {}'.format(
                    self.best_arms_set))

                if not self.is_leader:
                    self.phase = self.EXPLOIT
                elif self.is_leader:
                    # Compute the min empirical average reward from the best arms
                    self._update_best_arms_min_emp_rew()
                    if len(self.arms_to_add) == 0:
                        self.phase = self.EXPLOIT
                    else:
                        # Leader is starting communication again
                        self.t0 = self.state_round - 1
                        self.t1 = self.t0 + self.num_players - 1
                        self.t2 = self.t1 + self.num_players
                        self.t3 = self.t2 + self.K
                        self._add_log(
                            'Leader started communication (again): t0 {} t1 {} t2 {}'
                            .format(self.t0, self.t1, self.t2))
                        self.phase = self.COMMUNICATION
                        self.stats[self.START_OF_COMMUNICATION] += 1

    class CDPE(object):
        # Used for centralized coordination
        def __init__(self):
            self.players = None
            self.leader = None
            self.env = None

        def centralized(self):
            if not self.players or not self.leader:
                self._get_players()
            return self.players, self.leader

        def _get_players(self):
            if not self.env:
                raise Exception('Environment not provided to CDPE!')

            for p in self.env.players:
                if p.is_leader:
                    self.leader = p
                else:
                    if not self.players:
                        self.players = []
                    self.players.append(p)
            if len(self.players) == 0:
                self.players = None

        def update_env(self, env):
            self.env = env
            self.players = None
            self.leaders = None
            self._get_players()


class SynchComm(PlayerStrategy):
    """
    SIC MMAB
    """

    def __init__(self, narms, T=10, verbose=False):
        PlayerStrategy.__init__(self, narms, T)
        self.K0 = narms  # true number of arms (K used as number of active arms)
        self.name = 'SynchComm'
        self.ext_rank = -1  # -1 until known
        self.int_rank = 0  # starts index with 0 here
        self.M = 1  # number of active players
        self.T0 = np.ceil(
            self.K * np.e *
            np.log(T))  # length of Musical Chairs in initialization
        self.last_action = np.random.randint(
            self.K)  # last play for sequential hopping
        self.phase = 'fixation'
        self.t_phase = 0  # step in the current phase
        self.round_number = 0  # phase number of exploration phase
        self.active_arms = np.arange(0, self.K)
        self.sums = np.zeros(self.K)  # means*npulls
        self.last_phase_stats = np.zeros(self.K)
        self.verbose = verbose

    def play(self):
        """
        return arm to pull based on past information (given in update)
        """

        # Musical Chairs procedure in initialization
        if self.phase == 'fixation':
            if self.ext_rank == -1:  # still trying to fix to an arm
                return np.random.randint(self.K)
            else:  # fix
                return self.ext_rank

        # estimation of internal rank and number of players
        if self.phase == 'estimation':
            if self.t <= self.T0 + 2 * self.ext_rank:  # waiting its turn to sequential hop
                return self.ext_rank
            else:  # sequential hopping
                return (self.last_action + 1) % self.K

        # exploration phase
        if self.phase == 'exploration':
            last_index = np.where(self.active_arms == self.last_action)[0][0]
            return self.active_arms[(last_index + 1) %
                                    self.K]  # sequentially hop

        # communication phase
        if self.phase == 'communication':
            if (self.t_phase < (self.int_rank + 1) * (self.M - 1) * self.K *
                (self.round_number + 2)
                    and (self.t_phase >= (self.int_rank) *
                         (self.M - 1) * self.K * (self.round_number + 2))):
                # your turn to communicate
                # determine the number of the bit to send, the channel and the player

                t0 = self.t_phase % (
                    (self.M - 1) * self.K * (self.round_number + 2)
                )  # the actual time step in the communication phase (while giving info)
                b = (int)(
                    t0 %
                    (self.round_number + 2))  # the number of the bit to send

                k0 = (int)(((t0 - b) / (self.round_number + 2)) %
                           self.K)  # the arm to send
                k = self.active_arms[k0]
                if (((int)(self.last_phase_stats[k]) >> b) %
                        2):  # has to send bit 1
                    j = (t0 - b - (self.round_number + 2) * k0) / (
                        (self.round_number + 2) * self.K)  # the player to send
                    j = (int)(j + (j >= self.int_rank))
                    #print('Communicate bit {} about arm {} at player on arm {} by player {} at timestep {}'.format(b, k, self.active_arms[j], self.ext_rank, self.t_phase))
                    return self.active_arms[j]  # send 1
                else:
                    return self.active_arms[self.int_rank]  # send 0

            else:
                return self.active_arms[
                    self.int_rank]  # receive protocol or wait

        # exploitation phase
        if self.phase == 'exploitation':
            return self.last_action

    def update(self, play, obs):
        """
        Update the information, phase, etc. given the last round information
        X = obs[0]
        C = obs[1]
        """
        self.last_action = play

        if self.phase == 'fixation':
            if self.ext_rank == -1:
                if not (obs[1]):  # succesfully fixed during Musical Chairs
                    self.ext_rank = play

            # end of Musical Chairs
            if self.t == self.T0:
                self.phase = 'estimation'  # estimation of M
                self.last_action = self.ext_rank

        elif self.phase == 'estimation':
            if obs[1]:  # collision with a player
                if self.t <= self.T0 + 2 * self.ext_rank:  # increases the internal rank
                    self.int_rank += 1
                self.M += 1  # increases number of active players

            # end of initialization
            if self.t == self.T0 + 2 * self.K:
                self.phase = 'exploration'
                self.t_phase = 0
                self.round_number = (int)(
                    np.ceil(np.log2(self.M))
                )  # we actually not start at the phase p=1 to speed up the exploration, without changing the asymptotic regret

        elif self.phase == 'exploration':
            self.last_phase_stats[play] += obs[0]  # update stats
            self.sums[play] += obs[0]
            self.t_phase += 1

            # end of exploration phase
            if self.t_phase == (2 << self.round_number) * self.K:
                self.phase = 'communication'
                self.t_phase = 0

        elif self.phase == 'communication':
            # reception case
            if (self.t_phase >= (self.int_rank + 1) * (self.M - 1) * self.K *
                (self.round_number + 2)
                    or (self.t_phase < (self.int_rank) *
                        (self.M - 1) * self.K * (self.round_number + 2))):
                if obs[1]:
                    t0 = self.t_phase % (
                        (self.M - 1) * self.K * (self.round_number + 2)
                    )  # the actual time step in the communication phase (while giving info)
                    b = (int)(t0 % (self.round_number + 2)
                              )  # the number of the bit to send

                    k0 = (int)(((t0 - b) / (self.round_number + 2)) %
                               self.K)  # the channel to send
                    k = self.active_arms[k0]

                    self.sums[k] += ((2 << b) >> 1)

            self.t_phase += 1

            # end of the communication phase
            # update many things
            if (self.t_phase == (self.M) * (self.M - 1) * self.K *
                (self.round_number + 2) or self.M == 1):

                # update centralized number of pulls
                for k in self.active_arms:
                    self.npulls[k] += (2 << self.round_number) * self.M

                # update confidence intervals
                b_up = self.sums[self.active_arms] / self.npulls[
                    self.active_arms] + np.sqrt(
                        2 * np.log(self.T) / (self.npulls[self.active_arms]))
                b_low = self.sums[self.active_arms] / self.npulls[
                    self.active_arms] - np.sqrt(
                        2 * np.log(self.T) / (self.npulls[self.active_arms]))
                reject = []
                accept = []

                # compute the arms to accept/reject
                for i, k in enumerate(self.active_arms):
                    better = np.sum(b_low > (b_up[i]))
                    worse = np.sum(b_up < b_low[i])
                    if better >= self.M:
                        reject.append(k)
                        if self.verbose:
                            print(
                                'player {} rejected arm {} at round {}'.format(
                                    self.ext_rank, k, self.round_number))
                    if worse >= (self.K - self.M):
                        accept.append(k)
                        if self.verbose:
                            print(
                                'player {} accepted arm {} at round {}'.format(
                                    self.ext_rank, k, self.round_number))
                # update set of active arms
                for k in reject:
                    self.active_arms = np.setdiff1d(self.active_arms, k)
                for k in accept:
                    self.active_arms = np.setdiff1d(self.active_arms, k)

                # update number of active players and arms
                self.M -= len(accept)
                self.K -= (len(accept) + len(reject))

                if len(accept) > self.int_rank:  # start exploitation
                    self.phase = 'exploitation'
                    if self.verbose:
                        print('player {} starts exploiting arm {}'.format(
                            self.ext_rank, accept[self.int_rank]))
                    self.last_action = accept[self.int_rank]
                else:  # new exploration phase and update internal rank (old version of the algorithm where the internal rank was changed, but it does not change the results)
                    self.phase = 'exploration'
                    self.int_rank -= len(accept)
                    self.last_action = self.active_arms[
                        self.
                        int_rank]  # start new phase in an orthogonal setting
                    self.round_number += 1
                    self.last_phase_stats = np.zeros(self.K0)
                    self.t_phase = 0

        self.t += 1


class MCTopM(PlayerStrategy):
    """
    MCTopM strategy introduced by Besson and Kaufmann
    """

    def __init__(self, narms, M, T=10):
        PlayerStrategy.__init__(self, narms, T)
        self.name = 'MCTopM'
        self.last_action = np.random.randint(narms)
        self.C = False
        self.s = False
        self.bestM = np.arange(0, narms)
        self.M = M
        self.b = np.copy(self.B)
        self.previous_b = np.copy(self.B)

    def play(self):
        """
        return arm to pull
        """

        if self.last_action not in self.bestM:  # transition 3 or 5
            action = np.random.choice(
                np.intersect1d(
                    self.bestM,
                    np.nonzero(
                        self.previous_b <= self.previous_b[self.last_action])))
            self.s = False
        elif (self.C and not (self.s)):  # collision and not fixed
            action = np.random.choice(self.bestM)
            self.s = False
        else:  # tranistion 1 or 4
            action = self.last_action
            self.s = True

        return action

    def update(self, play, obs):
        self.last_action = play
        self.C = obs[1]
        self.t += 1
        self.means[play] = (self.npulls[play] * self.means[play] + obs[0]) / (
            self.npulls[play] + 1)
        self.npulls[play] += 1
        self.B[play] = np.sqrt(np.log(self.T) / (2 * self.npulls[play]))

        self.previous_b = np.copy(self.b)
        self.b = self.means + self.B
        self.bestM = np.argpartition(-self.b, self.M)[:self.M]


class MusicalChairs(PlayerStrategy):
    """
    Musical chairs strategy introduced by Rosenski et al.
    """

    def __init__(self, narms, T=10, delta=0.1):
        PlayerStrategy.__init__(self, narms, T)
        self.name = 'SynchComm'
        self.M = 1
        self.T0 = np.ceil(
            np.max([
                narms * np.log(2 * narms * narms * T) / 2,
                16 * narms * np.log(4 * narms * narms * T) / (delta * delta),
                narms * narms * np.log(2 * T) / 0.02
            ]))
        self.phase = 'exploration'
        self.fixed = -1
        self.bestM = None
        self.colls = 0

    def play(self):
        if self.phase == 'exploration':
            return np.random.randint(self.K)
        elif self.phase == 'fixation':
            return np.random.choice(self.bestM)
        else:
            return self.fixed

    def update(self, play, obs):
        self.t += 1
        if self.phase == 'exploration':
            if not (obs[1]):
                self.means[play] = (self.npulls[play] * self.means[play] +
                                    obs[0]) / (self.npulls[play] + 1)
                self.npulls[play] += 1
            else:
                self.colls += 1

            if self.t >= self.T0:
                self.phase = 'fixation'
                self.M = (int)(np.round(
                    np.log((self.t - self.colls) / self.t) /
                    (np.log(1 - 1 / self.K)))) + 1
                self.bestM = np.argpartition(-self.means, self.M)[:self.M]

        elif self.phase == 'fixation':
            if not (obs[1]):
                self.phase = 'exploitation'
                self.fixed = play
