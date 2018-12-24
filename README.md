# euler-diagram-drawer

## Описание

Программа получает на вход описание диаграммы Эйлера (например, `a b c ac bc`) и строит соответствующую диаграмму.

## Пример

```./qwe.py ab bc ac 2> /dev/null```

Это создаст два файла: `diagram_input.png` и `graph_input.png` с диаграммой и триангулированным двойственным графом, соответственно. Если запустить `./qwe.py` без параметров, будут созданы диаграммы для тестовых примеров в коде, например, `diagram_a_b_ab.png` и так далее.

## Зависимости

Для запуска нужен `python3`, а для корректной работы нужен модуль `pillow` для `python3`.

## Ссылки

- https://www.cs.kent.ac.uk/pubs/2008/2824/content.pdf
- https://arxiv.org/pdf/1201.3011.pdf
