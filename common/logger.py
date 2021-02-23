# -*- coding: utf-8 -*-


class Logger(object):
    def __init__(self, args):
        self.args = args

    def print_args(self):
        print("weight: ", self.args.weight)
        print("lr: ", self.args.lr)
        print("n_epoch: ", self.args.n_epoch)
        print("batch_size: ", self.args.batch_size)
        print("n_gen: ", self.args.n_gen)
        print("dataset: ", self.args.dataset)
        print("outdir: ", self.args.outdir)
        print("print_interval: ", self.args.print_interval)

    def print_log(self, epoch, it, train_loss, val_loss, a1, a5, ta1, ta5):
        print(f"epoch: {epoch}, iter: {it}, "
              f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}\n"
              f"train_acc@1 {a1:.3f}, train_acc@5 {a5:.3f}, "
              f"test_acc@1 {ta1:.3f}, test_acc@5 {ta5: .3f}")
