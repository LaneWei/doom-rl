class Model:
    """
    Base class for implementing a q_value based model that encapsulates a network
    (tensorflow graphs, keras model, etc.) and operations on this model.

    You need to implement the following methods:
        * `save_weights`
        * `load_weights`
        * `train`
        * `get_best_action`
        * `get_q_values` (optional)
        * `get_max_q_values` (optional)

    # Argument:
        * `preprocess_state_batch` A function that takes a batch of states as input,
        then perform pre-processing on the batch of states and return the processed batch.
    """
    def __init__(self, preprocess_state_batch=lambda x: x):
        self.preprocess_state_batch = preprocess_state_batch

    def save_weights(self, save_path):
        """
        Save the weights of this model to a file.

        :param save_path: The file path to save the weights of this model.
        """
        pass

    def load_weights(self, load_path):
        """
        Load the weights of this model from a file.

        :param load_path: The file path from which the weights of this model is to be loaded.
        """
        pass

    def train(self, state, action, target_q):
        """
        Perform a training step on this model.

        :param state: A batch of states.
        :param action: A batch of actions taken.
        :param target_q: Target q values.
        :return: The loss of this training step.
        """
        pass

    def get_best_action(self, state):
        """
        Given the current state as input, get the best action (with the highest q value)
        according to this model's network.

        :param state: The current state.
        :return: The the action with the highest q value.
        """
        pass

    def get_q_values(self, state):
        """
        Get all q values at a given state.
        :param state: The current state.
        :return: A list containing all q values.
        """
        raise NotImplementedError('Function get_q_values not implemented.')

    def get_max_q_values(self, state):
        """
        Get the highest q value at a given state.

        :param state: The current state.
        :return: The highest q value.
        """
        raise NotImplementedError('Function get_max_q_values not implemented.')
