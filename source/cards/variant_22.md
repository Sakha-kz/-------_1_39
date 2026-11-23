№22. Food-101 (10 классов) + EfficientNet-B0 + mixup
Суть: Food-101 subset 10, EfficientNet-B0 + mixup; фокус — mixup на текстурах
еды. Вопрос: mixup на текстурах еды: регуляризация или размывание
классов? : сравнить с №23 (cutmix) на семинаре.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1022. food101, отбери 10
классов по seed. EDA + contact sheet. 2. Классика (ст. 2). Лист обработки.
colorhist + LBP + LinearSVC: код блока 4, concat. Еда: цвет или микротекстура?
3. Аугментации (ст. 3). basic + mixup (--mixup 1.0). Абляция none vs mixup. 4.
Модель (ст. 4). efficientnet_b0 + ImageNet. Params/память. 5. Обучение (ст. 5).
Transfer probe→FT 10+10 (блок 9) с mixup, --norm imagenet --size 224. Кривые,
resume. 6. Оценка (ст. 6). Полная + robustness. 7. Grad-CAM (ст. 7). 5+5; объект
vs фон/текстура. 8. Перенос (ст. 8). transfer + FT с mixup: сравни probe vs FT,
влияние mixup на val. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (mixup:
регуляризация/размывание), таблица классика/scratch/transfer.

Команды:

python train.py --dataset food101 --model efficientnet_b0 --pretrained --aug basic --mixup
1.0 \
  --norm imagenet --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed
1022 --out_dir drive/out-var22-ft
python evaluate.py --ckpt drive/out-var22-ft/best.pt --dataset food101 --model efficientnet_b0
\
  --norm imagenet --size 224 --split val --seed 1022 --processing --gradcam --robustness --
latency --out_dir drive/eval-var22

Артефакты: абляция mixup, визуализация смешанных батчей, сравнение с
№23, concat-классика, отчёт. Ловушка: mixup применять к val/test — нельзя;
только train. Иначе метрики недействительны.

