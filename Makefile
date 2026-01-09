train-simplecnn:
	python train.py --cfg configs/simplecnn_cifar10.yaml

train-resnet18:
	python train.py --cfg configs/resnet18_cifar10.yaml

test-dataset-simplecnn:
	python test.py --cfg configs/simplecnn10_cifar10.yaml

test-dataset-resnet18:
	python test.py --cfg configs/resnet18_cifar10.yaml

test-gradio-simplecnn:
	python demo_gradio.py --cfg configs/simplecnn10_cifar10.yaml

test-gradio-resnet18:
	python demo_gradio.py --cfg configs/resnet18_cifar10.yaml
