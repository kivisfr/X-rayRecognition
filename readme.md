[***Язык: Русский***](#a-idtitle_ru-лабораторная-работа-a)

[***Language: English***](#a-idtitle_eng-laboratory-work-a)



# <a id="title_RU"> Лабораторная работа </a>
## Компьютерные методы статистического анализа данных 

### Задача и цель работы: 
Изучить и провести анализ предлагаемого датасета; написать программу нейросети, способной к обучению для распознавания проблемы, заложенной в датасете.

#### Датасет: Covid-19 X-ray - Two proposed Databases
#### [Ссылка на датасет](https://www.kaggle.com/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases)
#### [Научная статья автора датасета с его анализом](https://doi.org/10.3390/s21051742)
#### APA Style:
Vantaggiato, E., Paladini, E., Bougourzi, F., Distante, C., Hadid, A., & Taleb-Ahmed, A. (2021). COVID-19 Recognition Using Ensemble-CNNs in Two New Chest X-ray Databases. Sensors, 21(5), 1742. https://doi.org/10.3390/s21051742 


### Итоговое задание: 
Написание программы, которая использует машинное обучение, для распознавания и классификации рентгеновских снимков с высокой точностью по заданным категориям с учётом разделения датасета на количество рассматриваемых классов изображений.

## Подводные камни проекта
- данная версия рабочая, однако результаты даёт хуже, чем черновой вариант (он размещён с результатами в папке old_draft_version; только один summary.json исказился из-за ошибки);
- на данный момент в проекте продолжение обучения с чекпоинта не реализовано в полной мере: есть некоторые зачатки то тут, то там, однако это неполноценно работающая функция;
- сохранение каждой эпохи есть; объём занимаемой памяти разнился от 280 до 360 мб;
- есть ощущение, что модель ensemble в проекте выполнена паршиво (в черновом варианте всё выглядело адекватно);
- большая часть комментариев на английском; могут быть неточности.

### Обучение производилось на GPU: NVIDIA GeForce GTX 1060 6GB.


# <a id="title_ENG"> Laboratory work </a>
## Computer methods of statistical data analysis

### Objective and goal of the work:
To study and analyze the proposed dataset; to write a neural network program capable of learning to recognize the problem embedded in the dataset.

#### Dataset: Covid-19 X-ray - Two Proposed Databases
#### [Dataset link](https://www.kaggle.com/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases)
#### [Research article by the dataset's author with its analysis](https://doi.org/10.3390/s21051742)
#### APA Style:
Vantaggiato, E., Paladini, E., Bougourzi, F., Distante, C., Hadid, A., & Taleb-Ahmed, A. (2021). COVID-19 Recognition Using Ensemble-CNNs in Two New Chest X-ray Databases. Sensors, 21(5), 1742. https://doi.org/10.3390/s21051742

### Final assignment:
Write a program that uses machine learning to recognize and classify X-ray images with high accuracy into specified categories, taking into account the division of the dataset into the number of image classes being considered.

## Project Pitfalls
- This version is working, but the results are worse than the draft version (it's located with the results in the old_draft_version folder; only summary.json was corrupted due to an error);
- Continuing training from a checkpoint is not yet fully implemented in the project: there are some rudimentary features here and there, but it's not a fully functional function;
- Each epoch is saved; the memory footprint varied from 280 to 360 MB;
- It seems like the ensemble model in the project is poorly implemented (in the draft version, everything looked adequate);
- Most of the comments are in English; there may be inaccuracies.

### Training was performed on a NVIDIA GeForce GTX 1060 6GB GPU.