from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-rate', type=float, default=0, help='Dropout rate')
parser.add_argument('-epoch', type=int, default=50, help='Number of epochs')
args = parser.parse_args()

dropout_rate = args.rate
n_epochs = args.epoch

print(dropout_rate)
print(n_epochs)

