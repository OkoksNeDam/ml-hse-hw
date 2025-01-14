import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_sample_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака.
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    # Если у нас признаки имеют одинаковые значения, но при этом разные
    # таргеты, то возвращаем None.
    if len(np.unique(feature_vector)) == 1 and len(np.unique(target_vector)) > 1:
        return None, None, None, None
    
    sorted_indxs = feature_vector.argsort()
    # Сортируем вектор значений признака.
    feature_vector = feature_vector[sorted_indxs]
    target_vector = target_vector[sorted_indxs]

    # Получение списка отсортированных порогов, путем взятия среднего
    # соседних признаков.
    thresholds = list(map(lambda x: np.mean([x[0], x[1]]), zip(feature_vector, feature_vector[1:])))
    min_feature, max_feature = feature_vector[0], feature_vector[-1]

    # Убираем пороги, из-за которых могут быть пустые дочерние вершины.
    thresholds = list(filter(lambda x: x > min_feature and x < max_feature, thresholds))
    thresholds = np.unique(thresholds)

    ginis, left_nodes_size, right_nodes_size = [], [], []
    # Для каждого порога находим соответствующий критерий джини.
    for threshold in thresholds:
        left_node = target_vector[np.where(feature_vector < threshold)]
        right_node = target_vector[np.where(feature_vector >= threshold)]
        IG = calc_information_gain(target_vector, left_node, right_node)
        ginis += [IG]
        left_nodes_size += [len(left_node)]
        right_nodes_size += [len(right_node)]
    
    if min_sample_leaf:
        new_indicies = np.where((left_nodes_size >= min_sample_leaf) &
                                (right_nodes_size >= min_sample_leaf))[0]

        ginis = np.array(ginis)[new_indicies]
        if ginis.size == 0:
            return None, None, None, None
    
    best_threshold = thresholds[np.argmax(ginis)]
    best_gini = np.max(ginis)
    return thresholds, ginis, best_threshold, best_gini


def calc_information_gain(node, left, right):
    gini_left = calc_gini_criterion(left)
    gini_right = calc_gini_criterion(right)
    return (-gini_left * len(left) - gini_right * len(right)) / len(node)


def calc_gini_criterion(classes):
    if len(classes) == 0:
        return None
    # Количество объектов каждого класса.
    classes_count = np.unique(classes, return_counts=True)[1]
    # Доля объектов каждого класса.
    classes_shares = classes_count / len(classes)
    gini = 1 - np.sum(classes_shares ** 2)
    return gini


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        """
        :param feature_types: список типов для каждого признака. Принимает значения
        real или categorical.
        :param max_depth: максимальная глубина дерева.
        :param min_samples_split: минимальное количество элементов в вершине,
        для того, чтобы ее разбить.
        :param min_samples_leaf: минимальное количество элементов в листе.
        """
        # Тип каждого признака должен быть либо real, либо categorical.
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, curr_depth):
        node_gini = calc_gini_criterion(sub_y)
        # Проверяем условия разделения вершины.
        if node_gini is None or node_gini == 0.0 or \
         (self._max_depth and curr_depth == self._max_depth - 1) or \
         (self._min_samples_split and len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split, categories_split = None, None, None, None, None
        # Начинаем поиск лучшего признака для разбиения вершины.
        for feature in range(sub_X.shape[1]):
            # Тип текущего признака.
            feature_type = self._feature_types[feature]
            categories_map = {}

            # Если признак вещественный, то просто вытаскиваем его из данных.
            if feature_type == "real":
                feature_vector = np.array(list(map(float, sub_X[:, feature])))
            # Если признак категориальный, то используем специальный алгоритм
            # для кодирования.
            elif feature_type == 'categorical':
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = dict(sorted(ratio.items(), key=lambda item: item[1]))
                categories_map = dict(zip(sorted_categories.keys(), list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            # Находим лучший порог и соответствующий критерий gini 
            # для текущего признака.
            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            # Нашли порог, который оказался самым лучшим по gini.
            if threshold and gini and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == 'categorical':
                    categories_split = categories_map

                threshold_best = threshold

        # Если gini не вырос при лучшем разбиении, то будем считать
        # вершину листовой.
        if feature_best is None or gini_best + node_gini < 0:
            node["type"] = "terminal"
            # Берем самый популярный класс из вершины.
            node["class"] = Counter(sub_y).most_common()[0][0]
            return

        node["type"] = "nonterminal"

        # Признак, по которому будем разбивать вершину.
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best

        # Если признак категориальный, то добавляем словарь, с помощью
        # которого будем кодировать новые признаки.
        if self._feature_types[feature_best] == "categorical":
            node["categories_split"] = categories_split

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], curr_depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], curr_depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        categories_split = node.get('categories_split')
        feature_value = x[node['feature_split']]
        # Если признак категориальный, то кодируем ему с помощью
        # сохраненного при обучении словаря categories_split.
        if categories_split is not None:
            feature_value = categories_split.get(x[node['feature_split']], 0)
        if feature_value < node['threshold']:
            return self._predict_node(x, node['left_child'])
        else:
            return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
