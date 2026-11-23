№35. CIFAR-10 + long-tail
Суть: CIFAR-10 искусственный long-tail, ResNet-18; фокус — что лечит
дисбаланс. Вопрос: что лечит дисбаланс: данные, loss или и то и другое
(macro-F1)? : deferred re-weighting (DRW) — четвёртый метод.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1035. cifar10, построй long-
tail: блок 3 long_tail_indices(targets, ratio=100, seed), подставь в train-сплит. EDA:
гистограмма «хвоста». 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC
(balanced): код блока 4, balanced=True. class_weight vs focal loss у CNN. 3.
Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4). resnet18.
Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 30 эпох. Кривые, resume.
6. Оценка (ст. 6). Полная + robustness; главная метрика — macro-F1, per-class
F1. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). 3 прогона:
weighted loss / oversampling (WeightedRandomSampler) / focal loss (напиши по
arXiv:1708.02002); per-class F1 до/после. Бонус — DRW. 9. Отчёт (ст. 9). Latency,
cost, ответ на вопрос (данные/loss), таблица классика/scratch/imbalance.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1035 --out_dir drive/out-var35
python evaluate.py --ckpt drive/out-var35/best.pt --dataset cifar10 --model resnet18 \
 --split val --seed 1035 --processing --gradcam --robustness --latency --out_dir drive/eval-
var35

Артефакты: гистограмма long-tail, таблица 3 методов (macro-F1/per-class), focal
loss код, отчёт. Ловушка: оценивать long-tail по accuracy — редкие классы
«тонут»; только macro-F1 и per-class F1.

