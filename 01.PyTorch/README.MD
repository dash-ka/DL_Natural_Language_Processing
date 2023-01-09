# 1. PyTorch and Neural Networks

This repo contains materials that introduce you **PyTorch** library, очень мощному и современному фреймворку, на котором создаются нейронные сети. В конце занятия мы познакомимся с первым видом нейронных сетей -- **Multi-Layer Perceptron (MLP)**, или **Fully-Connected Network (FC)**, по-русски **Полносвязные нейросети**. 

<p align=center>
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width=400>
</p>

### Лекция: Многослойные нейросети

Чтобы узнать общую информацию о многослойных сетях и о том, как с их помощью решать проблему многоклассовой классификации, посмотрите [**запись лекции**](https://www.youtube.com/watch?v=FSb5vuG6Bkw).

### PyTorch: Basics

Существует много разных фреймворков для создания нейросетей (например, **TensorFlow**), но в нашем курсе мы будем использовать **PyTorch**, потому что он намного быстрее развивается и в целом проще в освоении. Его API очень похож на API NumPy, поэтому Вам будет легче, если вы уже понимаете принципы работы NumPy и знаете его основные функции.

Предлагаем посмотреть [**запись семинара**](https://www.youtube.com/watch?v=UoBYayBPh84) и выполнить задания в [**[seminar]pytorch_basics.ipynb**](./[seminar]pytorch_basics.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1Wrur4sqHVO0MQLX0-9ifBsD79HXWUtUF). 

При изучении таких больших фреймворков может оказаться полезной [шпаргалка по основным функциям](https://pytorch.org/tutorials/beginner/ptcheat.html), которая находится на официальном сайте PyTorch.

### PyTorch: Building NNs

На прошлом занятии вы создавали свой нейрон с помощью NumPy, теперь пришло время сделать то же самое, но с помощью PyTorch. Jupyter-ноутбук [**[seminar]neuron_pytorch.ipynb**](./[seminar]neuron_pytorch.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1rKTMDU_zvhqTM7FosQOTd1Nb3Aw-eaD9) поможет Вам в этом, а в записи семинара (ссыка выше) семинарист уже разбирает, как ее полностью выполнить. Выполнив тетрадь, вы можете сравнить, насколько отличаются реализации на NumPy и на PyTorch.

### Practice: Yelp Reviews Classification :)

Пришло время написать свою первую настоящую нейросеть - полносвязную сеть (еще называют Fully Connected, или Dense, как и ее лиейные слои). Как всегда, вы можете использовать jupyter-ноутбук [**[Sentiment Classification with.ipynb**](./Sentiment Classification with NNs.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://drive.google.com/open?id=1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB) с пропущенным кодом, смотря запись семинара в случае трудностей.

