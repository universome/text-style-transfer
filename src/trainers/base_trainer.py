from tqdm import tqdm; tqdm.monitor_interval = 0


class BaseTrainer:
    def __init__(self, config):
        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 10)
        self.validate_every = config.get('validate_every')
        self.plot_every = config.get('plot_every')

    def run_training(self, training_data, val_data=None, plot_every=50, val_bleu_every=100):
        should_continue = True

        self.train_mode()

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(training_data, leave=False):
                    self.train_step(batch, val_data)
            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

    def train_step(self, batch, val_data):
        self.train_on_batch(batch)

        if self.validate_every and self.num_iters_done % self.validate_every == 0:
            self.validate(val_data)

        if self.plot_every and self.num_iters_done % self.plot_every == 0:
            self.plot_scores()

        self.num_iters_done += 1

    def train_on_batch(self, batch):
        pass

    def validate(self, val_data):
        pass

    def plot_scores(self):
        pass

    def train_mode(self):
        pass
