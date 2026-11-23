№34. CIFAR-10 + регуляризация
Суть: CIFAR-10, ResNet-18, 4 регуляризатора; фокус — какой эффективнее на
50k. Вопрос: какой регуляризатор эффективнее на 50k изображений и
почему? : комбинация двух лучших — синергия или нет.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1034. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC, sweep C ∈ {0.1, 1,
10}: код блока 4, цикл по C. Регуляризация в SVM vs CNN. 3. Аугментации (ст.
3). basic. Абляция none vs basic. 4. Модель (ст. 4). resnet18. 4 конфигурации:
baseline / dropout / label smoothing (--label_smoothing 0.1) / BN→GroupNorm
(replace_bn_with_gn, блок 5). Diff приложи. 5. Обучение (ст. 5). SGD lr 0.1 cosine,
30 эпох × 4. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; таблица acc
по регуляризаторам, train/val gap. 7. Grad-CAM (ст. 7). 5+5 для лучшего; объект
vs фон. 8. Перенос (ст. 8). Лучшая конфигурация → transfer: probe→FT (блок
9). 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (какой регуляризатор), sweep
C классики, таблица классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --label_smoothing 0.1 --
optimizer sgd \
 --lr 0.1 --scheduler cosine --epochs 30 --seed 1034 --out_dir drive/out-var34-ls
python evaluate.py --ckpt drive/out-var34-ls/best.pt --dataset cifar10 --model resnet18 \
 --split val --seed 1034 --processing --gradcam --robustness --latency --out_dir drive/eval-
var34

Артефакты: diff модификаций, таблица 4 регуляризаторов, sweep C (код),
отчёт. Ловушка: sweep C через --classical не сделать (фиксированный C=1) —
нужен код блока 4 с циклом.

