№39. Capstone: максимальный пайплайн ★
Суть: CIFAR-10/100, EfficientNet-B0 или ViT-tiny transfer; фокус — максимум
accuracy при бюджете T4 30 минут. Вопрос: максимум accuracy при бюджете
T4 30 минут — что даёт каждый компонент (абляционная таблица)? :
сравнить с №1 и разобрать вклад каждого улучшения.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1039. cifar10 (или cifar100).
EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG + SIFT-BoW +
colorhist (concat) + LinearSVC: код блока 4, concat трёх признаков. Полный
классический стек vs DL-ансамбль. 3. Аугментации (ст. 3). strong + mixup (--aug
strong --mixup 1.0). Абляция none vs strong+mixup. 4. Модель (ст. 4). efficientnet_b0
или vit_tiny + ImageNet. Params/память. 5. Обучение (ст. 5). AdamW lr 3e-4 +
cosine + label smoothing 0.1 + AMP (--amp), transfer probe→FT (блок 9), --norm
imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6). Полный набор + TTA (блок
11 tta_predict) + ensemble из 2 моделей; финальный test-замер один раз. 7. Grad-
CAM (ст. 7). 5+5 (attention rollout для ViT); объект vs фон. 8. Перенос (ст. 8).
transfer + FT + TTA: абляционная таблица вклада каждого компонента
(аугментации/mixup/TTA/ensemble/AMP). 9. Отчёт (ст. 9). Latency, cost (бюджет
30 мин T4), ответ на вопрос (вклад компонентов), сравнение с №1 (бонус),
таблица классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model efficientnet_b0 --pretrained --aug strong --mixup
1.0 \
  --label_smoothing 0.1 --amp --norm imagenet --size 224 --optimizer adamw --lr 3e-4 \
  --scheduler cosine --epochs 10 --seed 1039 --out_dir drive/out-var39-ft
python evaluate.py --ckpt drive/out-var39-ft/best.pt --dataset cifar10 --model efficientnet_b0
\
  --norm imagenet --size 224 --split val --seed 1039 --processing --gradcam --robustness --
latency --out_dir drive/eval-var39

Артефакты: абляционная таблица компонентов, TTA+ensemble код, concat-
классика, сравнение с №1, отчёт. Ловушка: включить все компоненты сразу
без абляции — нельзя ответить, что именно дало выигрыш; добавляй по
одному.

