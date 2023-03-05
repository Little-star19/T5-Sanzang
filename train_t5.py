import os
import argparse
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from utils import *
from transformers.models.t5 import T5ForConditionalGeneration
import warnings

warnings.filterwarnings('ignore')


class TaskLightModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = model.forward(input_ids=batch['decoder_input_ids'], labels=batch['decoder_input_ids'])
        loss, logits = outputs[:2]
        # logits = self(**batch).logits
        # loss = ce_loss(logits, batch['labels'], batch['decoder_attention_mask'])

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.args.lr, self.args.weight_decay)
        if self.args.max_epochs == -1:
            t_total = self.args.max_steps // self.args.accumulate_grad_batches
        else:
            t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        if self.args.warmup_steps != -1:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        return [optimizer],[{"scheduler": scheduler, "interval":"step"}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ========================= Train and trainer ==========================
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--eval_start', default=3, type=int)
    parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--plugins', type=str, default='ddp_sharded')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kfold', type=int, default=1)


    # ========================= Data ==========================
    parser.add_argument('--train_file', type=str, default='data.txt', required=False)
    parser.add_argument('--dev_file', type=str, default='data.txt', required=False)
    parser.add_argument('--predict_file', type=str, required=False)
    parser.add_argument('--noise_prob', default=0., type=float)
    parser.add_argument('--mlm_probability', default=0.15, type=float)
    parser.add_argument('--max_source_length', default=512, type=int)
    parser.add_argument('--max_target_length', default=200, type=int)
    parser.add_argument('--beams', default=3, type=int)
    parser.add_argument('--num_works', type=int, default=4)

    # ========================= Model ==========================
    parser.add_argument('--model_path', type=str, default='./mt5')
    parser.add_argument('--save_path', type=str, default='./saved')

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    tokenizer = T5PegasusTokenizer.from_pretrained(args.model_path)
    data = EncoderDecoderData(args, tokenizer)
    # print(data.read_file('news2016zh_1.txt'))  # 读取数据
    dataloaders = data.get_dataloader()
    # l = tokenizer.tokenize('我爱[SEP]中国')
    # print(l)
    # print(tokenizer.convert_tokens_to_ids(l))
    # q = tokenizer.convert_tokens_to_ids(l)
    # print(tokenizer.convert_ids_to_tokens([100,101,102,103,1234]))
    # print(tokenizer.get_special_tokens_mask([100,101,102,103,1234],already_has_special_tokens=True))
    print(tokenizer.convert_tokens_to_ids(tokenizer.mask_token))

    for fold in range(args.kfold):
        pl.seed_everything(args.seed + fold)
        train_data, dev_data = dataloaders['train'][fold], dataloaders['dev'][fold]
        model = TaskLightModel(args)
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=args.save_path,
            filename='t5_copy-noise={}-{}-'.format(args.noise_prob,
                                                   fold) + "{epoch:02d}-{bleu:.4f}-{rouge-1:.4f}-{rouge-2:.4f}-{rouge-l:.4f}",
            save_weights_only=True,
            save_on_train_epoch_end=True,
            monitor='loss',
            mode='min',
        )
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=False)
        trainer.fit(model, train_data, dev_data)
        del model
        del trainer
        torch.cuda.empty_cache()















