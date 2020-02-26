import numpy as np


class FullSensingMultiPlayerMAB(object):
    """
    Structure of stochastic MAB in the full sensing model (adapted to both Collision and Statistic Sensing settings)
    """

    def __init__(self,
                 means,
                 nplayers,
                 strategy,
                 reward='Bernoulli',
                 graph=None,
                 **kwargs):
        self.K = len(means)
        np.random.shuffle(means)
        self.means = np.array(means)
        self.M = nplayers
        self.graph = graph
        self.reward_type = reward
        self.players = [
            strategy(narms=self.K, **kwargs) for _ in range(nplayers)
        ]  # l of all players and their strategy

        # If graph model is enabled, set relative id and communicatinon interface
        if graph:
            for i in range(len(self.players)):
                id_player = graph.relative_id(i)
                player_intf = graph.communication_interface(id_player)
                self.players[i].set_graph_mode(
                    relative_id=id_player, interface=player_intf)

    def simulate_single_step_rewards(self):
        if self.reward_type == 'Gaussian':
            return np.random.normal(self.means, 30)
        return np.random.binomial(1, self.means)

    def simulate_single_step(self, plays):
        """
        return to each player its stat and collision indicator where plays is the vector of plays by the players
        """
        rews = self.simulate_single_step_rewards()
        if not self.graph:
            unique, counts = np.unique(
                plays,
                return_counts=True)  # compute the number of pulls per arm
            # remove the collisions
            collisions = unique[counts > 1]  # arms where collisions happen
            cols = np.array([p in collisions for p in plays
                             ])  # the value is 1 if there is collision
            # generate the stats X_k(t)
            rewards = rews[plays] * (1 - cols)
        else:
            cols = self.graph.simulate_collisions()
            # @todo code for reward * cols
            rewards = rews[plays]
        return list(zip(rews[plays], cols)), rewards

    def simulate(self, horizon=None, exit_condition=None):
        """
        Return the vector of regret for each time step until horizon
        """

        rewards = []
        play_history = []

        T = horizon

        for t in range(T):
            plays = np.zeros(self.M)
            plays = [(int)(player.play())
                     for player in self.players]  # plays of all players

            obs, rews = self.simulate_single_step(
                plays)  # observations of all players

            for i in range(self.M):
                self.players[i].update(
                    plays[i], obs[i])  # update strategies of all player

            if self.graph:
                self.graph.propagate()

            rewards.append(np.sum(rews))  # list of rewards
            play_history.append(plays)
            if exit_condition is not None:
                if exit_condition(self.players) == True:
                    T = t + 1
                    break
        if not self.graph:
            top_means = -np.partition(-self.means, self.M)[:self.M]
        else:
            top_means = np.max(self.means) * self.M
        best_case_reward = np.sum(top_means) * np.arange(1, T + 1)
        cumulated_reward = np.cumsum(rewards)

        regret = best_case_reward - cumulated_reward
        self.regret = (regret, best_case_reward, cumulated_reward)
        self.top_means = top_means
        return regret, play_history

    def get_players(self):
        return self.players


class GraphModelMAB(object):
    class CommunicationInterface(object):
        def __init__(self, id_player):
            self.msg_to_send = None
            self.msg_to_recv = []
            self._id = id_player

        def send(self, timestamp, arm):
            self.msg_to_send = (timestamp, arm)

        def receive(self):
            a = [i for i in self.msg_to_recv]
            self.msg_to_recv = []
            return a

        def is_empty(self):
            return len(self.msg_to_recv) == 0

        @property
        def id(self):
            return self._id

    def __init__(self, communication_graph, collision_graph, ids):
        # Collision graph not implemented yet
        # Does not support directed graph
        self._collision_graph = None
        self._communication_graph = communication_graph
        self._relative_ids = ids

        if 0 not in self._relative_ids:
            raise Exception(
                'Id 0 not present in graph! Needed to define a leader.')

        self._communication_graph_size = (len(communication_graph),
                                          len(communication_graph[0]))
        if self._communication_graph_size[0] != self._communication_graph_size[
                1]:
            raise Exception(
                'Communication graph adjacency matrix is not square!')

        self._build_neighbors()
        self.channel = {i: self.CommunicationInterface(i) for i in ids}

    def _build_neighbors(self):
        self._neighbors = {
            self._relative_ids[i]: []
            for i in range(self._communication_graph_size[0])
        }
        for i in range(len(self._communication_graph)):
            for j in range(self._communication_graph_size[0]):
                if self._communication_graph[i][j] == 1:
                    self._neighbors[self._relative_ids[i]].append(
                        self._relative_ids[j])

    def simulate_collisions(self):
        a = [0 for i in self._relative_ids]
        return a

    def propagate(self):
        for idp, interface in self.channel.items():
            msg = interface.msg_to_send
            if msg is not None:
                for neighbor in self._neighbors[idp]:
                    self.channel[neighbor].msg_to_recv.append(msg)

    def relative_id(self, i):
        return self._relative_ids[i]

    def communication_interface(self, id_player):
        return self.channel[self._relative_ids.index(id_player)]
