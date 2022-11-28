# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from learningAgents import ReinforcementAgent
import random, util


DEFAULT_VALUE = 0.0
DEFAULT_ACTION = None

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q = {}
        self.epsilon = float(args['epsilon'])
        self.alpha = float(args['alpha'])
        self.gamma = float(args['gamma'])

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if state not in self.Q or action not in self.Q[state]:
            return DEFAULT_VALUE
        return self.Q[state][action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if state not in self.Q or not self.Q[state]:
            return DEFAULT_VALUE
        possible_actions = self.getLegalActions(state)
        possible_values = [self.Q[state][action] if action in self.Q[state] else DEFAULT_VALUE
                           for action in possible_actions]
        return max(possible_values)


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if state not in self.Q or not self.Q[state]:
            possible_actions = self.getLegalActions(state)

        else:
            max_value = self.computeValueFromQValues(state)
            possible_actions = [action
                                for action in self.Q[state]
                                if self.Q[state][action] == max_value]

        if not possible_actions:
            return DEFAULT_ACTION

        return random.choice(possible_actions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        if not self.getLegalActions(state):
            return None

        action = ""

        if self.computeActionFromQValues(state) is None:
            action = self.actionExplore(state)

        elif util.flipCoin(self.epsilon):
            action = self.actionExplore(state)

        else:
            action = self.actionExploit(state)

        return action

    def actionExplore(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        return random.choice(actions)

    def actionExploit(self, state):
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = DEFAULT_VALUE
        current_q = self.Q[state][action]
        dq = self.alpha * (reward + (self.gamma * self.computeValueFromQValues(nextState))
                           - current_q)
        self.Q[state][action] = current_q + dq

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def __str__(self):
        return str({"epsilon": self.epsilon,
                    "gamma": self.gamma,
                    "alpha": self.alpha})
