import json
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from rectools import Columns


class KionReco:
    """
    Класс, содержащий методы получения рекомендаций по датасету Kion
    """

    def __init__(self, model_name_, dataset_):
        assert Path(
            model_name_).is_file()  # проверка на наличие файла с моделью
        assert Path(
            dataset_).is_file()  # проверка на наличие файла с датасетом
        # подгружаем модель
        with open(Path(model_name_), 'rb') as f:
            self.model = dill.load(f)
        # подгружаем датасет
        with open(Path(dataset_), 'rb') as f:
            self.dataset = dill.load(f)

        self.sorted_top = self.dataset.interactions.df[
            [Columns.Item]].value_counts().reset_index()[Columns.Item].values

    def check_user(self, user_id) -> bool:
        return False if len(np.where(
            self.dataset.user_id_map.external_ids == user_id)[0]) == 0 \
            else True

    def reco_recommend(self, user_id, k_recos=10) -> np.ndarray:
        """
        Получение К рекомендаций для пользователя
        :param user_id: идентификатор пользователя
        :param k_recos: количество рекомендаций
        :return:
        """
        if self.check_user(user_id):
            # рекомендации для теплого пользователя (который попал в обучение)
            df_recos = self.model.recommend(
                users=[user_id],
                dataset=self.dataset,
                k=k_recos,
                filter_viewed=True
            )
            return df_recos[Columns.Item].values
        else:
            return self.sorted_top[:k_recos]

    def reco(self, user_id, k_recos=10) -> np.ndarray:
        """
        Получение К рекомендаций для пользователя
        :param user_id: идентификатор пользователя
        :param k_recos: количество рекомендаций
        :return:
        """
        if self.check_user(user_id):
            # рекомендации для теплого пользователя (который попал в обучение)
            df_recos = self.model.predict(
                users=[user_id],
                dataset=self.dataset,
                k=k_recos,
                filter_viewed=True
            )
            return df_recos
        else:
            return self.sorted_top[:k_recos]


class KionRecoBM25(KionReco):
    def __init__(self, model_name_, dataset_):
        super().__init__(model_name_, dataset_)
        cnt = Counter(self.dataset.interactions.df['item_id'].values)
        self.idf = pd.DataFrame.from_dict(cnt, orient='index',
                                          columns=['doc_freq']).reset_index()
        n = self.dataset.interactions.df.shape[0]
        self.idf['idf'] = self.idf['doc_freq'].apply(
            lambda x: np.log((1 + n) / (1 + x) + 1))

        self.users_inv_mapping = dict(
            enumerate(self.dataset.interactions.df['user_id'].unique()))

        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(
            enumerate(self.dataset.interactions.df['item_id'].unique()))

        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

        watched = self.dataset.interactions.df.groupby('user_id').agg(
            {'item_id': list})
        self.watched = {k: v['item_id'] for k, v in
                        json.loads(watched.T.to_json()).items()}
        self.mapper = self.generate_implicit_recs_mapper()

    def generate_implicit_recs_mapper(self):
        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            recs = self.model.similar_items(user_id, N=50)
            users = [self.users_inv_mapping[user] for user, _ in recs]
            sims = [sim for _, sim in recs]
            return users, sims

        return _recs_mapper

    def make_reco_slow(self, user_id, k_recos=10) -> np.ndarray:
        recs = pd.DataFrame({
            'user_id': self.dataset.interactions.df[
                self.dataset.interactions.df['user_id'] == user_id][
                'user_id'].unique()
        })
        recs['similar_user_id'], recs['similarity'] = zip(
            *recs['user_id'].map(self.mapper))

        # explode lists to get vertical representation
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        # delete recommendations of itself
        recs = recs[~(recs['user_id'] == recs['similar_user_id'])]

        #     # join watched items
        recs = recs.merge(self.watched, left_on=['similar_user_id'],
                          right_on=['user_id'], how='left')
        recs = recs.explode('item_id')
        # drop duplicates pairs user_id-item_id
        # keep with the largest similiarity
        recs = recs.sort_values(['user_id', 'similarity'], ascending=False)
        recs = recs \
            .merge(
            self.idf[['index', 'idf']],
            left_on='item_id',
            right_on='index',
            how='left') \
            .drop(['index'], axis=1)
        recs['rank_idf'] = recs['similarity'] * recs['idf']
        recs = recs.sort_values(['user_id', 'rank_idf'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1
        return recs[recs['rank'] <= k_recos]['item_id'].values

    def make_reco(self, user_id, k_recos=10):
        try:
            recss = {}
            # находим близких пользователей
            recss['similar_user_id'], recss['similarity'] = self.mapper(user_id)

            # удаляем самого себя
            recss['similar_user_id'] = recss['similar_user_id'][1:]
            recss['similarity'] = recss['similarity'][1:]

            # извлекаем просмотренные фильмы близких пользователей
            recss['item_id'] = [self.watched.get(f"{x}") for x in
                                recss['similar_user_id']]

            # объединяем с idf
            recss = pd.DataFrame(recss).explode('item_id').sort_values(
                ['similarity'], ascending=False)
            recss = recss.merge(self.idf[['index', 'idf']],
                                left_on='item_id',
                                right_on='index',
                                how='left').drop(['index'], axis=1)
            recss['rank_idf'] = recss['similarity'] * recss['idf']
            recss = recss.sort_values(['rank_idf'], ascending=False)
            recss.dropna(inplace=True)
            recos = recss['item_id'].unique()[:k_recos]

            # если рекомендаций меньше
            if len(recos) < k_recos:
                recos = pd.DataFrame(np.append(recos, self.sorted_top),
                                     columns=['recos'])['recos'].unique()[:k_recos]
        except:
            recos = self.sorted_top[:k_recos]
        return recos

    def reco(self, user_id, k_recos=10) -> np.ndarray:
        """
        Получение К рекомендаций для пользователя
        :param user_id: идентификатор пользователя
        :param k_recos: количество рекомендаций
        :return:
        """
        if self.check_user(user_id):
            # рекомендации для теплого пользователя (который попал в обучение)
            df_recos = self.make_reco(user_id, k_recos)
            return df_recos
        else:
            return self.sorted_top[:k_recos]
