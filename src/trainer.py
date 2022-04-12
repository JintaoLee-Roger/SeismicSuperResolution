from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, 
                my_loss=None, my_lossv=None, ckp=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # ckp: checkpoint
        self.loader_train = loader.loader_train # 
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.lossv = my_lossv
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for batch, (lr, hr, _, _1) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.model.eval()
        if not self.args.test_only:
            self.lossv.start_log()

        scale = self.args.scale
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for (lr, hr, filename, params) in tqdm(self.loader_test, ncols=80):
            lr, hr = self.prepare(lr, hr)
            sr = self.model(lr)
            if not self.args.test_only:
                lossv = self.lossv(sr, hr)

            save_list = [sr]
            if not self.args.apply_field_data:
                self.ckp.log[-1] += utility.calc_psnr(
                    sr, hr, scale
                )

            if self.args.save_results:
                self.ckp.save_results(filename[0], save_list, params)

        if not self.args.apply_field_data:
            self.ckp.log[-1] /= len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                'Seis', 
                scale,
                self.ckp.log[-1],
                best[0],
                best[1] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if not self.args.test_only:
            self.lossv.end_log(len(self.loader_test))

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        return [a.to(device) for a in args]

    def terminate(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs

