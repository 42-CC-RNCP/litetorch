class Callback:
    """
    Base class for all callbacks.
    train -> epoch -> batch -> batch -> epoch -> train
    """

    def __init__(self):
        pass
    
    def on_train_begin(self, trainer):
        """
        Called at the beginning of training.
        """
        pass

    def on_epoch_begin(self, trainer):
        """
        Called at the beginning of each epoch.
        """
        pass

    def on_batch_begin(self, trainer):
        """
        Called at the beginning of each batch.
        """
        pass

    def on_batch_end(self, trainer):
        """
        Called at the end of each batch.
        """
        pass
    
    def on_epoch_end(self, trainer):
        """
        Called at the end of each epoch.
        """
        pass
    
    def on_train_end(self, trainer):
        """
        Called at the end of training.
        """
        pass
