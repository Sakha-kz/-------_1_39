№33. CIFAR-10 + sweep расписаний LR
Суть: CIFAR-10, ResNet-18, 3 расписания LR; фокус — влияние расписания на
качество/сходимость. Вопрос: как расписание влияет на финальное качество
и скорость сходимости? : warmup-абляция к лучшему расписанию.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1033. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Точка отсчёта: классика без расписания. 3. Аугментации (ст. 3). basic.
Абляция none vs basic. 4. Модель (ст. 4). resnet18. Params/память. 5. Обучение
(ст. 5). 3 прогона: --scheduler step / cosine / onecycle, SGD lr 0.1, 30 эпох. Кривые lr,
resume. 6. Оценка (ст. 6). Полная + robustness; таблица acc/скорость
сходимости по расписаниям. 7. Grad-CAM (ст. 7). 5+5 для лучшего; объект vs
фон. 8. Перенос (ст. 8). Лучшее расписание → FT: transfer probe→FT (блок 9). 9.
Отчёт (ст. 9). Latency, cost, ответ на вопрос (расписание), warmup-абляция
(бонус), таблица классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --optimizer sgd --lr 0.1 \
  --scheduler onecycle --epochs 30 --seed 1033 --out_dir drive/out-var33-onecycle
python evaluate.py --ckpt drive/out-var33-onecycle/best.pt --dataset cifar10 --model resnet18
\
  --split val --seed 1033 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var33

Артефакты: кривые lr + acc для 3 расписаний, warmup-абляция, отчёт.
Ловушка: onecycle требует lr как пик и шагов по батчам — не переноси
настройки cosine механически.

