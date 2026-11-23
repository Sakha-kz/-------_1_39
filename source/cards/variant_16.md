№16. DTD текстуры + SmallCNN
Суть: DTD (47 текстур), SmallCNN; фокус — текстуры без глобальной формы,
роль GAP. Вопрос: текстуры без глобальной формы: помогает ли GAP и какие
признаки выучиваются? : Grad-CAM — равномерное ли внимание по
текстуре.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1016. dtd (47 классов,
128×128). EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. LBP +
LinearSVC: --classical lbp. Текстуры — родная стихия классики? 3. Аугментации
(ст. 3). Random crop разных масштабов (правь блок 2: T.RandomResizedCrop).
Абляция none vs policy. 4. Модель (ст. 4). smallcnn с GAP. Params/память;
абляция GAP vs flatten (часть вопроса). 5. Обучение (ст. 5). SGD lr 0.1 cosine, 30
эпох. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness. 7. Grad-CAM (ст.
7). 5+5; равномерное ли внимание по текстуре (бонус). 8. Перенос (ст. 8).
domain shift → синтетические паттерны: оцени на сгенерированных
текстурах, дообучи; кривые адаптации. 9. Отчёт (ст. 9). Latency, cost, ответ на
вопрос (GAP/признаки), таблица классика/scratch/transfer.

Команды:

python train.py --dataset dtd --model smallcnn --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1016 --out_dir drive/out-var16
python evaluate.py --ckpt drive/out-var16/best.pt --dataset dtd --model smallcnn \
 --split val --seed 1016 --classical lbp --processing --gradcam --robustness --latency --out_dir
drive/eval-var16

Артефакты: абляция GAP vs flatten, Grad-CAM текстур, таблица LBP, отчёт.
Ловушка: для текстур flatten вместо GAP ломает инвариантность к позиции
— это и есть предмет вопроса, не «исправляй» молча.

