№17. EuroSAT + ResNet-18 transfer
Суть: EuroSAT (спутники, 10 классов), ResNet-18 transfer; фокус — перенос
между сенсорами/масштабами. Вопрос: спутниковые снимки: перенос между
сенсорами/масштабами? : per-class анализ (какие типы местности сложнее).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1017. eurosat (10 классов,
64×64, офиц. test нет → 80/10/10). EDA + contact sheet. 2. Классика (ст. 2). Лист
обработки. colorhist + HOG + LinearSVC: код блока 4, concat. Цвет vs структура
местности. 3. Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст.
4). resnet18 + ImageNet. Params/память. 5. Обучение (ст. 5). Transfer probe→FT
10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6).
Полная + robustness; per-class анализ (бонус). 7. Grad-CAM (ст. 7). 5+5; объект

vs фон. 8. Перенос (ст. 8). domain shift → UC Merced: оцени на ucmerced,
дообучи; кривые forgetting/адаптации между сенсорами. 9. Отчёт (ст. 9).
Latency, cost, ответ на вопрос (перенос сенсоров), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset eurosat --model resnet18 --pretrained --aug basic --norm imagenet
\
  --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1017 --out_dir
drive/out-var17-ft
python evaluate.py --ckpt drive/out-var17-ft/best.pt --dataset eurosat --model resnet18 \
  --norm imagenet --size 224 --split val --seed 1017 --processing --gradcam --robustness --
latency --out_dir drive/eval-var17

Артефакты: per-class анализ, кривые domain shift EuroSAT→UC Merced, concat-
классика, отчёт. Ловушка: UC Merced скачивается вручную (ImageFolder
root/ucmerced/train|test/<класс>) — без него стадия 8 не выполнить.

