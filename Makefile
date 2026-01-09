train-simplecnn:
	python -m src.train.train --cfg configs/simplecnn10_cifar10.yaml

train-resnet18:
	python -m src.train.train --cfg configs/resnet18_cifar10.yaml

test-dataset-simplecnn:
	python -m src.test.test_dataset --cfg configs/simplecnn10_cifar10.yaml

test-gradio-simplecnn:
	python -m src.test.test_gradio --cfg configs/simplecnn10_cifar10.yaml

test-dataset-resnet18:
	python -m src.test.test_dataset --cfg configs/resnet18_cifar10.yaml

test-gradio-resnet18:
	python -m src.test.test_gradio --cfg configs/resnet18_cifar10.yaml
