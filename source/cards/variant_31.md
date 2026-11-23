№31. CIFAR-10 + абляция аугментаций
Суть: CIFAR-10, SmallCNN, 4 политики аугментаций; фокус — вклад каждого
уровня. Вопрос: сколько accuracy даёт каждый уровень аугментаций и где
насыщение? : время обучения vs выигрыш — compute-эффективность
аугментаций.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1031. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Фиксированная точка отсчёта для абляции. 3. Аугментации (ст. 3). 4
прогона: none / basic / strong / strong+cutmix (--aug strong --cutmix 1.0). Одинаковые
модель/seed/эпохи. 4. Модель (ст. 4). smallcnn. Params/память. 5. Обучение (ст.
5). SGD lr 0.1 cosine, 30 эпох × 4. Кривые, resume для каждого. 6. Оценка (ст. 6).
Полная + robustness; таблица Δ val acc по политикам, кривые насыщения. 7.
Grad-CAM (ст. 7). 5+5 для лучшей политики; объект vs фон. 8. Перенос (ст. 8).
Лучшая политика → transfer ResNet-18: probe→FT (блок 9), --norm imagenet --
size 224. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (насыщение), compute-
эффективность (бонус), таблица классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model smallcnn --aug strong --cutmix 1.0 --optimizer sgd \
 --lr 0.1 --scheduler cosine --epochs 30 --seed 1031 --out_dir drive/out-var31-strong-cutmix
python evaluate.py --ckpt drive/out-var31-strong-cutmix/best.pt --dataset cifar10 --model
smallcnn \
 --split val --seed 1031 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var31

Артефакты: таблица 4 политик (Δ acc + время), кривые, отчёт. Ловушка:
менять seed/эпохи между политиками — абляция недействительна; всё
одинаковое, кроме аугментаций.

