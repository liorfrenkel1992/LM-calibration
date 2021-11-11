import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import argparse

def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet110'
    save_loc = './'
    save_plots_loc = './'
    saved_model_name = 'resnet110_cross_entropy_350.model'
    num_bins = 35
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'
    trained_loss = 'cross_entropy'
    logits_path = '/mnt/dsi_vol1/users/frenkel2/data/calibration/trained_models/spline/logits/'
    #logits_path = 'C:/Users/liorf/OneDrive - Bar-Ilan University/calibration/trained_models/spline/logits/'
    logits_file = 'probs_resnet110_c10_logits.p'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    parser.add_argument("-acc", action="store_true", dest="acc_check",
                        help="whether to calculate ECE for each class only if accuracy gets better")
    parser.add_argument("-iters", type=int, default=1,
                        dest="temp_opt_iters", help="number of temprature scaling iterations")
    parser.add_argument("-init_temp", type=float, default=2.5,
                        dest="init_temp", help="initial temperature for temperature scaling")
    parser.add_argument("-const_temp", action="store_true", dest="const_temp",
                        help="whether to use constant temperature on all classes")
    parser.add_argument("--save-path-plots", type=str, default=save_plots_loc,
                        dest="save_plots_loc",
                        help='Path to save plots')
    parser.add_argument("--loss", type=str, default=trained_loss,
                        dest="trained_loss",
                        help='Trained loss(cross_entropy/focal_loss/focal_loss_adaptive/mmce/mmce_weighted/brier_score)')
    parser.add_argument("--logits_path", type=str, default=logits_path,
                        dest="logits_path",
                        help='Path of saved logits')
    parser.add_argument("--logits_file", type=str, default=logits_file,
                        dest="logits_file",
                        help='File of saved logits')
    parser.add_argument("-bins", action="store_true", dest="bins_temp",
                        help="whether to calculate ECE for each bin separately")
    parser.add_argument("-dists", action="store_true", dest="dists",
                        help="whether to optimize ECE by dists from uniform probability")
    parser.add_argument("--divide", type=str, default="equal_divide", dest="divide",
                        help="How to divide bins (reg/equal)")

    return parser.parse_args()


gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("Today is a nice day", return_tensors="pt").input_ids

generated_outputs = gpt2.generate(input_ids, do_sample=True, num_return_sequences=3, output_scores=True)

# only use id's that were generated
# gen_sequences has shape [3, 15]
gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

# let's stack the logits generated at each step to a tensor and transform
# logits to probs
probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

# now we need to collect the probability of the generated token
# we need to add a dummy dim in the end to make gather work
gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

# now we can do all kinds of things with the probs

# 1) the probs that exactly those sequences are generated again
# those are normally going to be very small
unique_prob_per_sequence = gen_probs.prod(-1)

# 2) normalize the probs over the three sequences
normed_gen_probs = gen_probs / gen_probs.sum(0)
assert normed_gen_probs[:, 0].sum() == 1.0, "probs should be normalized"

# 3) compare normalized probs to each other like in 1)
unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)

if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()