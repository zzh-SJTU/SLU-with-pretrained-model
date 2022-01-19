import sys, os, time, gc
from torch.optim import Adam
import json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
Example.configuration(args.dataroot)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path,istest=1)
test_dataset = Example.load_dataset(test_path,istest=1)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: test -> %d" % (len(test_dataset)))
args.pad_idx = 0
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
model.load_state_dict(torch.load("model.bin", map_location=device)['model'])


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode():
    model.eval()
    dataset = test_dataset
    predictions, labels = [], []
    args.batch_size=1
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            ex_list = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, ex_list, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            if label is not None:
                labels.extend(label)
            if loss is not None:
                total_loss += loss
            count += 1
        #metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return predictions, total_loss / count



start_time = time.time()
predictions,  loss = decode()

with open(os.path.join(args.dataroot, 'test_unlabelled.json'), 'r', encoding='utf-8') as fi:
    test_set = json.load(fi)
    points = [point for x in test_set for point in x]
for point, pred in zip(points, predictions):
    point['pred'] = [p.split('-') for p in pred]

with open(os.path.join(args.dataroot, 'test.json'), 'w', encoding='utf-8') as fo:
    json.dump(test_set, fo, indent=4, ensure_ascii=False)