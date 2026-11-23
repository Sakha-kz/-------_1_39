№38. CIFAR-10 + self-supervised ★
Суть: CIFAR-10, SSL-предобучение → linear probe; фокус — экономия меток.
Вопрос: сколько меток экономит self-supervised предобучение (accuracy vs
число меток)? : кривая probe accuracy при 10/50/100% меток.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1038. cifar10; unlabeled train
для SSL. EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG +
LinearSVC: --classical hog. Точка отсчёта кривой label-efficiency. 3. Аугментации
(ст. 3). strong (для SSL — контрастные пары). Абляция none vs strong. 4. Модель
(ст. 4). Backbone smallcnn/resnet18. Params/память. 5. Обучение (ст. 5). SSL:
SimCLR-lite или rotation prediction (блок 11, реализация твоя) на unlabeled →
linear probe. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; сравни с
supervised (№1). 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). SSL
→ linear probe → сравнение с supervised: кривая probe acc при 10/50/100%
меток (бонус). 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (экономия меток),
кривая label-efficiency, таблица классика/supervised/SSL.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug strong --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1038 --out_dir drive/out-var38-supervised
python evaluate.py --ckpt drive/out-var38-supervised/best.pt --dataset cifar10 --model
resnet18 \

 --split val --seed 1038 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var38

Артефакты: код SSL, кривая label-efficiency (10/50/100%), сравнение с
supervised, отчёт. Ловушка: SSL-probe оценивать на тех же метках, что и SSL-
предобучение — утечка; probe только на отдельном labeled subset.

