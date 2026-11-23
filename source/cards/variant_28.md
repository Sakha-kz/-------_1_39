№28. UTKFace (возрастные группы) + ResNet-18
Суть: UTKFace (возрастные группы), ResNet-18; фокус — Grad-CAM + этика/bias.
Вопрос: какие признаки модель использует для возраста; есть ли перекосы
по подгруппам? : раздел «Этические ограничения» в отчёте (риски, bias,
применения).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1028. utkface (вручную;
возрастные группы собираются из имён файлов — скрипт парсинга
приложи сам, это часть задания). EDA + contact sheet. 2. Классика (ст. 2). Лист
обработки. HOG + LinearSVC: --classical hog. Bias ручных признаков vs CNN. 3.
Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4). resnet18 +
ImageNet. Params/память. 5. Обучение (ст. 5). Transfer probe→FT 10+10 (блок 9),
--norm imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness;
per-class/подгруппы — есть ли перекосы. 7. Grad-CAM (ст. 7). 5+5; какие
признаки модель использует для возраста. 8. Перенос (ст. 8).
Сбалансированное дообучение: weighted loss/oversampling по подгруппам;
per-class F1 до/после. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос
(признаки/bias), раздел «Этические ограничения» (бонус), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset utkface --model resnet18 --pretrained --aug basic --norm imagenet
\
  --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1028 --out_dir
drive/out-var28-ft
python evaluate.py --ckpt drive/out-var28-ft/best.pt --dataset utkface --model resnet18 \
  --norm imagenet --size 224 --split val --seed 1028 --classical hog --processing --gradcam --
robustness --latency --out_dir drive/eval-var28

Артефакты: скрипт парсинга возраста, анализ перекосов по подгруппам,
этический раздел, отчёт. Ловушка: без собственного скрипта парсинга имён
файлов возрастные группы не собрать — это обязательная часть стадии 1.

