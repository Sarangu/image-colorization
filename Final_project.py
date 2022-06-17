import preprocess
import train
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Task', dest='task', required=True,
                        choices=["Preprocess", "Train", "Test"],
                        help='Loss function to use'),
    parser.add_argument('--train', dest='train', required=False,
                        help='Full path to the raw training zip file')
    parser.add_argument('--preprocessed_train', dest='preprocessed_train', required=False,
                        help='Full path to the training file after preprocessing')
    parser.add_argument('--epochs', dest='epochs', required=False,
                        help='Number of epochs you want to train the model for')
    parser.add_argument('--loss_function', dest='loss', required=False,
                        choices=["mse", "2DCrossEntropy"],
                        help='Loss function to use'),
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='Batch size for training'),
    parser.add_argument('--model_path', dest='model_path', required=False,
                        help='The full path to the directory where you want to save the model.')
    parser.add_argument('--test_image_path', dest='test_image_path', required=False,
                        help='The full path to the test image file.')
    parser.add_argument('--test_image_path_predicted', dest='test_image_path_predicted', required=False,
                        help='The full path to the predcited test image file.')
    args = parser.parse_args()
    
    if args.task == "Preprocess":
        preprocess.preprocess(args.train, args.preprocessed_train)
        
    elif args.task == "Train":
#         preprocess.preprocess(args.train, args.preprocessed_train)
        training_object = train.train()
        training_object.train_model_with_args(args.preprocessed_train, int(args.epochs), args.loss, int(args.batch_size), args.model_path)
    elif args.task == "Test":
        training_object = train.train()
        training_object.convert_image_to_color(args.test_image_path, args.test_image_path_predicted, args.model_path)