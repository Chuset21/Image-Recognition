from imageai.Prediction import ImagePrediction
import os


def print_predictions(prediction: ImagePrediction, execution_path: str, image_name: str) -> None:
    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, image_name), result_count=5)
    print(f'{image_name}:')
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(f'{eachPrediction}  :  {eachProbability}')
    print()


def main():
    execution_path = os.getcwd()

    prediction: ImagePrediction = ImagePrediction()
    prediction.setModelTypeAsSqueezeNet()
    prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()

    for image in ('giraffe.jpg', 'house.jpg', 'godzilla.jpg'):
        print_predictions(prediction, execution_path, image)


if __name__ == '__main__':
    main()
