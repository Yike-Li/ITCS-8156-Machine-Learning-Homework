import time
import random
import numpy as np
import pandas as pd
from copy import copy
from functools import reduce
import matplotlib.pyplot as plt
from collections import defaultdict

SUIT = ['H', 'S', 'D', 'C']
RANK = ['A', '2', '3', '4', '5', '6', '7']
RANK_VALUE = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 10, 'Q': 10,
              'K': 10}


class Card:
    def __init__(self, rank, suit):
        self.rank = rank  # ['A', '2', '3', '4', '5', '6', '7']
        self.suit = suit  # ['H','S','D','C']
        self.rank_to_val = RANK_VALUE[self.rank]  # map rank to score values

    def __str__(self):  # gives a string representation of the object, when using print or str. 打印出对用户友好的信息。这里是打印牌信息
        return f"{self.rank}{self.suit}"

    def __repr__(self):  # 不管直接输出对象还是通过print打印的信息都按我们__repr__方法中定义的格式进行显示了
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):  # 判断两张牌是否相同。相同return True，不同return False。
        return self.rank == other.rank and self.suit == other.suit


# Deck class contains some basic operations performed with the cards:
# 1. Shuffling the cards.
# 2. Drawing card from the deck.
class Deck:
    def __init__(self, packs):
        self.packs = packs  # packs是有几副牌。当为1时，只有一副
        self.cards = []  # 一开始没牌。下面紧接着发牌
        for pack in range(0, packs):  # 若packs为1，生成一副牌。每副有4种花色，13个rank.这样生成的一副牌是按顺序的，用前需shuffle
            for suit in SUIT:  # ['H','S','D','C']
                for rank in RANK:  # ['A', '2', '3', ..., '7']
                    self.cards.append(Card(rank, suit))  # 都是Card class的object，存到cards这个list里

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        card = self.cards[0]  # 抽第一张牌
        self.cards.pop(0)  # 去除cards里的第一张牌，因为已经抽过
        return card


class Player:
    """
    Player class to create a player object.
    eg: player = Player("player1", list(), isBot = False)
    Above declaration will be for your agent.
    All the player names should be unique or else you will get error.

    """

    # list() function creates a list object。
    def __init__(self, name, stash=list(), isBot=False, points=0, conn=None):
        self.stash = stash  # Stash represents the hand of the Player
        self.name = name  # player名字
        self.game = None
        self.isBot = isBot  # 是否是电脑
        self.points = points  # 分数初始化为0
        self.conn = conn  # 有无server connection

    def deal_card(self, card):
        """
        Deal a Card to the Player 给玩家发牌:
        appends the card in the stash and check the condition that length of stash should not be greater than
        the number of cards length in game
        Args:
             card:  The Card object provided to Player as part of the deal
        Returns:
             No returns
        """
        try:
            self.stash.append(card)
            if len(self.stash) > self.game.cardsLength + 1:  # cardsLength???
                raise ValueError('Cannot have cards greater than ')
        except ValueError as err:
            print(err.args)

    def drop_card(self, card):
        self.stash.remove(card)  # remove the card from stash list
        self.game.add_pile(card)  # 把弃的牌加到pile最上面，pile是给对方亮的牌
        return -1

    def meld(self):
        """
            This method tries to find the cards with the same rank in the hand.
            If it finds then it will merge the cards in the hand to the melded cards array in the game.
        """
        # defualtdict: 跟dict很像但never raises a KeyError。如果给出一个非dict里的key，会返回空值，不会出KeyError
        # It provides a default value for the key that does not exists.
        card_hash = defaultdict(list)  # create hash table，如果给非dict里key返回空list
        for card in self.stash:  # 遍历手中的牌
            # 手中牌以list形式存card_hash里，以牌rank为key. 如果两张牌rank相同，则可以meld合并
            card_hash[card.rank].append(card)

        melded_card_ranks = []
        # card_hash.items()： dict_items([(2, ['2H', '2D']), (5, ['5S']),...])
        for (card_rank, meld_cards) in card_hash.items():
            if len(meld_cards) >= 3:  # 如果其中一个key下element长度>=3，则有3张同rank的牌
                self.game.meld.append(meld_cards)  # 存放这三张一样的牌
                melded_card_ranks.append(card_rank)  # 存放三张一样的牌的rank
                for card in meld_cards:
                    self.stash.remove(card)  # 从player手中去除这三张一样的牌

        for card_rank in melded_card_ranks:  # 当len(meld_cards) < 3, melded_card_ranks为空。只有meld时下面才执行
            card_hash.pop(card_rank)  # 从card_hash里去除这三张一样的牌

        # 若melded_card_ranks不为空，返回True，表meld成功游戏结束
        return len(melded_card_ranks) > 0

    def stash_score(self):
        score = 0  # 初始化
        for card in self.stash:
            score += RANK_VALUE[card.rank]  # 统计手中牌的score
        return score

    def get_info(self, debug):
        """
        This function fetch all the information of the player.
        """
        if debug:
            print(
                f'Player Name : {self.name} \nStash Score: {self.stash_score()} \nStash : {", ".join(str(x) for x in self.stash)}')
        card_ranks = []
        card_suits = []
        pileset = None
        pile = None
        for card in self.stash:
            card_suits.append(RANK_VALUE[card.rank])  # 集合所有手上牌的card  number
            card_ranks.append(card.suit)  # 集合所有手上牌的card suit
        # 这里的pile应该是历史的pile总集。这里有没有pile，return的元素不同。这句有啥用？
        if len(self.game.pile) > 0:
            return {"Stash Score": self.stash_score(),
                    "CardSuit": card_suits, "CardRanks": card_ranks,
                    "PileSuit": self.game.pile[-1].suit,  # 显示最后一张pile的suit
                    "PileRank": self.game.pile[-1].rank}  # 显示最后一张pile的rank
        return {"Stash Score": self.stash_score(), "CardSuit": card_suits, "CardRanks": card_ranks}


class RummyAgent():
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Simple Rummy Environment
    ---------------------------------------------------------------------------------------------------------------------------
    Simple Rummy is a game where you need to make all the cards in your hand same before your opponent does.
    Here you are given 3 cards in your hand/stash to play.
    For the first move you have to pick a card from the deck or from the pile. 
    The card in deck would be random but you can see the card from the pile.
    In the next move you will have to drop a card from your hand.
    Your goal is to collect all the cards of the same rank. 
    Higher the rank of the card, the higher points you lose in the game. 
    You need to keep the stash score low. Eg, if you can AH, 7S, 5D your strategy would be 
        to either find the first pair of the card or by removing the highest card in the deck.
    You only have 20 turns to either win the same or collect low scoring card.
    You can't see other players cards or their stash scores.
    ===========================================================================================================================
    Parameters
    ---------------------------------------------------------------------------------------------------------------------------
    players: Player objects which will play the game.
    max_card_length : Number of cards each player can have
    max_turns: Number of turns in a rummy game

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def __init__(self, players, max_card_length=5, max_turns=20):
        self.max_card_length = max_card_length
        self.max_turns = max_turns
        self.reset(players)  # 清空players等所有信息，创建一副牌。这里相当于执行了reset里一堆初始化

    def reset(self, players, max_turns=20):  # reinitialize all the deck, pile and players
        self.players = []
        self.deck = Deck(1)  # 生成一副牌
        self.deck.shuffle()
        self.meld = []  # 初始化
        self.pile = []  # 初始化
        self.max_turns = max_turns  # 初始化
        self.update_player_cards(players)  # 储存player信息并给每个player抓牌

    def update_player_cards(self, players):
        for player in players:  # 把每个player变成Player object
            player = Player(player.name, list(), isBot=player.isBot,
                            points=player.points, conn=player.conn)
            stash = []
            for i in range(self.max_card_length):
                player.stash.append(self.deck.draw_card())  # 给每个player摸够3张牌
            # 把player(为Player object)里的game属性令为当前这个Rummy game
            player.game = self
            self.players.append(player)  # 把这些个player的object都存到self.players
        self.pile = [self.deck.draw_card()]  # 从洗好的deck中取最上面的一张牌作为pile

    def add_pile(self, card):
        """
        This method takes a card as argument and first checks number of cards in the deck.
        If its is ‘0’ then add the cards from pile to deck and append the passed card to the pile.
        """

        if len(self.deck.cards) == 0:  # 如果deck没牌了
            self.deck.cards.extend(self.pile)  # 把pile的所有牌加到deck
            self.deck.shuffle()  # 洗牌
            self.pile = []  # 清空pile
        self.pile.append(card)  # 把上个player弃的牌加到pile上

    def pick_card(self, player, action):
        """
        This methods helps player picking up the card from either pile or deck based on action.
        """
        if action == 0:
            self.pick_from_pile(player)  # 如果action为0，拿pile的牌
        else:
            self.pick_from_deck(player)  # 否则（action=1）则从deck里抽取
        if player.meld():  # 如果返回为True，玩家能meld, 胜利。下面reward数字可以改！
            return {"reward": 10}
        else:  # 如果为False，玩家不能meld，扣reward
            return {"reward": -1}
            # return -player.stash_score()

    def pick_from_pile(self, player):
        """
        This method helps player picking card from the pile and simultaneously a card from pile gets reduced.
        """
        card = self.pile[-1]  # 从pile取牌
        self.pile.pop()  # 从pile中去掉这张牌
        return player.stash.append(card)

    def pick_from_deck(self, player):
        return player.stash.append(self.deck.draw_card())  # 从deck随机抽牌

    def get_player(self, player_name):  # fetch the details of the player given player_name
        return_player = [
            player for player in self.players if player.name == player_name]
        if len(return_player) != 1:
            print("Invalid Player")
            return None
        else:
            return return_player[0]

    def drop_card(self, player, card):  # 玩家弃一张牌，并相应获得reward
        player.drop_card(card)
        return {"reward": -1}

    def computer_play(self, player):
        """
        defines the play of the computer/Dealer in following sequence:
        --> Randomly taking actions from picking up card from deck/pile.
        --> Checking the meld condition afterwards.
        --> If the meld condition does not satisfy, randomly remove the card from his stash.
        """
        # Gets a card from deck or pile
        if random.randint(0, 1) == 1:  # 0和1随机选一个
            self.pick_from_pile(player)
        else:
            self.pick_from_deck(player)

        # tries to meld if it can
        # if random.randint(0,10) > 5 :
        player.meld()  # 这里update了stash

        # 随机removes a card from the stash
        if len(player.stash) != 0:
            card = player.stash[(random.randint(0, len(player.stash) - 1))]
            player.drop_card(card)

    def play(self):
        """
        defines all the function city of play for the player
      --> Decrementing the maximum number of turns defined per game.
      --> For each player, it will check the 'stash', if the 'Stash' for any player = 0 (That player won), it will add the value of each card in stash for all other players.
      --> Or If maximum number of turns in each round becomes 0, it will add the card values in stash for all the players and return.
        """
        for player in self.players:
            # if any player stash is 0: "he won" and sum stash for other players
            if len(player.stash) == 0:
                return True
        if self.max_turns <= 0:  # if max no. of turns<=0, game done and sum stash
            return True
        return False

    def _update_turn(self):  # count the number of turns in the game
        self.max_turns -= 1


class RLAgent:
    """
    Yike's RL Agent Model for Rummy
    """

    def __init__(self, env):  # env is a RummyAgent object
        self.rummy = env
        self.RANK_VALUE = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                           'T': 10, 'J': 10, 'Q': 10, 'K': 10}
        self.n_pick_a = 2  # no. of pick actions
        self.n_drop_a = 4  # no. of drop actions
        # 3张牌各7种可能性（不管花色只看rank），7,7,7,7+1(1因为可能pile里没牌),2
        self.Q_pick = np.zeros((7, 7, 7, 7, self.n_pick_a))
        # 4张牌各7种可能性（只看rank），7,7,7,7,4(四种弃牌可能性)
        self.Q_drop = np.zeros((7, 7, 7, 7, self.n_drop_a))

        self.scores_end_game = []  # stores non-bot stash scores at each end game
        self.rewards_per_game = []  # stores the rewards gained per game by the non-bot player
        self.avg_reward_per_action = []  # stores the avg rewards gained per turn by the non-bot player
        self.rewards = []  # stores the rewards gained by pick/drop within a turn

    def epsilon_pick(self, epsilon, s):
        if np.random.uniform() < epsilon:
            a = np.random.randint(self.n_pick_a)  # randomly pick one from the available pick actions
        else:  # 这里用index-1是因为index从0到6，如果来了个state是7，则会超出范围
            i_max = np.where(self.Q_pick[s[0] - 1, s[1] - 1, s[2] - 1, s[3] - 1, :]  # 返回所有等于最大值的index
                             == np.max(self.Q_pick[s[0] - 1, s[1] - 1, s[2] - 1, s[3] - 1, :]))[0]
            a = int(np.random.choice(i_max))  # 从所有最大当中随机选一个action作为选中的action
        return a

    def epsilon_drop(self, epsilon, s):
        if np.random.uniform() < epsilon:
            a = np.random.randint(self.n_drop_a)
        else:
            i_max = np.where(self.Q_drop[s[0] - 1, s[1] - 1, s[2] - 1, s[3] - 1, :]
                             == np.max(self.Q_drop[s[0] - 1, s[1] - 1, s[2] - 1, s[3] - 1, :]))[0]
            a = int(np.random.choice(i_max))
        return a

    def train(self, **params):  # training steps to update Q table
        # parameters
        gamma = params.pop('gamma', 0.9)  # discount rate
        alpha = params.pop('alpha', 0.1)  # learning rate
        epsilon = params.pop('epsilon', 0.1)  # epsilon greedy - 10% random select action
        maxiter = params.pop('maxiter', 1000)  # 1000场比赛
        reset = params.pop('reset', False)  # reset the 2 Q tables
        debug = params.pop('debug', False)  # debug mode

        # record time starts
        start_time = time.time()

        # init self.Q matices
        if reset:
            self.Q_pick[...] = 0
            self.Q_drop[...] = 0

        # init scores_end_game and rewards_per_game each time we run the train method
        self.scores_end_game = []  # stores non-bot stash scores at each end game
        self.rewards_per_game = []  # stores the rewards gained per game by the non-bot player
        self.avg_reward_per_action = []  # stores the avg rewards gained per turn by the non-bot player

        for j in range(maxiter):  # 循环1000场比赛
            for player in self.rummy.players:
                player.points = player.stash_score()  # 初始化stash score

            self.rummy.reset(self.rummy.players)  # reinitialize deck, pile and players, 给每个player抓牌
            random.shuffle(self.rummy.players)  # shuffle player的出牌顺序

            self.rewards = []  # stores rewards of each turn from pick/drop for the non-bot player

            if debug:
                print(f'*************** Game {j + 1} starts (total={maxiter}) *************** ')

            while not self.rummy.play():  # False：如果还没有player赢, 或max_turns>0
                if debug:
                    print(f'********** Turn {20 + 1 - self.rummy.max_turns} starts (total=20) *********     ')
                self.rummy._update_turn()  # max_turns-1
                for player in self.rummy.players:  # loop each player
                    # for the computer player
                    if player.isBot:
                        # print('Bot plays...')
                        if self.rummy.play():  # True: 如果有player赢了
                            continue  # 跳过当前循环的剩余语句 (跳过这个player)，然后继续进行下一轮循环(下个player)
                        # if debug:
                        #     print(f'{player.name} Plays...')
                        self.rummy.computer_play(player)  # 机器执行摸牌弃牌动作
                        if debug:  # fetch all information of the player
                            player.get_info(debug)  # gives player name, stash score, and stash
                        if player.stash == 0:  # 如果stash里没牌了；这里是执行摸牌弃牌动作后的判断
                            print(f'{player.name} wins the round')

                    # for the non-bot player, we train Q table
                    else:
                        if self.rummy.play():  # True: 如果有player赢了
                            continue  # 跳出当前循环

                        # returns player cards info in dict. debug=False disables the print.
                        player_info = player.get_info(debug=debug)

                        # We only care about rank of the cards. Below build the state based on 'CardSuit' in play_info
                        pick_state = []
                        for rank in player_info['CardSuit'][:3]:  # CardSuit stores the rank of cards...
                            pick_state.append(rank)  # append rank of cards to the pick state
                        if 'PileRank' in player_info:  # if pile is available
                            pick_state.append(
                                self.RANK_VALUE[player_info['PileRank']])  # use RANK_VALUE to convert 'A' to 1
                        else:  # if the pile is not available (this is when a bot wins and we are to sum the stash score)
                            pick_state.append(0)  # append 0 at the 4th card place. When sum, we only care the first 3
                        if debug:
                            print(f'State before picking: {pick_state}')

                        # pick card and gather reward
                        pick_a = self.epsilon_pick(epsilon, pick_state)  # the pick action chosen based on state
                        if debug:
                            print(f'pick action is {pick_a}')  # action==0，拿pile; action==1，拿deck

                        pick_reward = self.rummy.pick_card(player, pick_a)  # get reward from pick action
                        if debug:
                            print(f'Card Pick Reward: {pick_reward}')
                        adj_pick_reward = pick_reward['reward'] - player.stash_score()  # this is the adj reward
                        self.rewards.append(adj_pick_reward)  # 存下pick action的reward

                        if len(player.stash) == 1:  # 如果执行pick_card后手里只剩1张牌，说明meld成功，玩家胜利，只用当前状态更新Q
                            self.Q_pick[pick_state[0] - 1,
                                        pick_state[1] - 1,
                                        pick_state[2] - 1,
                                        pick_state[3] - 1, pick_a] += alpha * pick_reward['reward']

                        elif len(player.stash) >= 3:  # 如果执行pick_card后手里还是有>=3张牌，则游戏继续
                            pick_state_ = player.get_info(debug=debug)['CardSuit']  # gain state after pick action
                            if debug:
                                print(f'State after pick: {pick_state_}')

                            # pick_a_ = self.epsilon_pick(epsilon,
                            #                             pick_state_)  # find next pick action using next pick state
                            # update Q_pick table using SARSA
                            self.Q_pick[pick_state[0] - 1,
                                        pick_state[1] - 1,
                                        pick_state[2] - 1,
                                        pick_state[3] - 1, pick_a] += \
                                alpha * (adj_pick_reward + gamma * np.max(self.Q_pick[pick_state_[0] - 1,
                                                                               pick_state_[1] - 1,
                                                                               pick_state_[2] - 1,
                                                                               pick_state_[3] - 1, :]) -
                                         self.Q_pick[
                                             pick_state[0] - 1,
                                             pick_state[1] - 1,
                                             pick_state[2] - 1,
                                             pick_state[3] - 1, pick_a])

                        if debug:
                            print(f'.***** ***** Yike pick complete. ***** *****.')

                        # dropping a card
                        if len(player.stash) == 1:  # 如果玩家手里只有一张牌，意味着赢了，下面是弃掉最后那张牌
                            self.rummy.drop_card(player, player.stash[0])  # drop the only card in stash, win the game
                            if debug:
                                print(f'{player.name} Wins the round')

                        elif len(player.stash) != 0:  # 如果手里牌不为1也不为0, 执行常规弃牌步骤
                            drop_state = player.get_info(debug)['CardSuit']  # 得到当前手中牌的state，4张
                            if debug:
                                print(f'State before drop: {drop_state}')

                            drop_a = self.epsilon_drop(epsilon, drop_state)  # 4张选一张来drop

                            drop_card = player.stash[drop_a]
                            if debug:
                                print(f'Dropped Card: {drop_card}')
                            drop_reward = self.rummy.drop_card(player, drop_card)  # conduct drop action
                            if debug:
                                print(f'Drop Reward: {drop_reward}')
                            adj_drop_reward = drop_reward['reward'] - player.stash_score()
                            self.rewards.append(adj_drop_reward)  # 存下drop action的reward

                            if debug:
                                print(player.get_info(debug=False))

                            drop_state_ = player.get_info(debug=debug)['CardSuit']  # gain rank states after drop action
                            if debug:
                                print(f'State after drop: {drop_state_}')
                            drop_state_.append(1)  # append a 1 to the 4th spot, then 1-1 is 0 (for Q_drop update)
                            # drop_a_ = self.epsilon_drop(epsilon, drop_state_)

                            self.Q_drop[drop_state[0] - 1,  # update self.Q_drop tables (SARSA)
                                        drop_state[1] - 1,
                                        drop_state[2] - 1,
                                        drop_state[3] - 1, drop_a] += \
                                alpha * (adj_drop_reward + gamma * np.max(self.Q_drop[drop_state_[0] - 1,
                                                                               drop_state_[1] - 1,
                                                                               drop_state_[2] - 1,
                                                                               drop_state_[3] - 1, :]) -
                                         self.Q_drop[drop_state[0] - 1,
                                                     drop_state[1] - 1,
                                                     drop_state[2] - 1,
                                                     drop_state[3] - 1, drop_a])
                        else:  # 如果手里牌数为0
                            if debug:
                                print(f'***** {player.name} Wins the round *****')
                        if debug:
                            player.get_info(debug)  # 看下drop后的player state情况

            if len(self.rewards) > 0:  # 如果rewards里有元素
                self.rewards_per_game.append(np.sum(self.rewards))  # sum the rewards gained by pick/drop
                self.avg_reward_per_action.append(np.mean(self.rewards))  # the avg rewards gained per action

            else:  # 如果rewards里没有元素(bot player上了就赢了)
                for player in self.rummy.players:
                    player.points = player.stash_score()
                    if player.name == 'Yike':
                        self.rewards_per_game.append(- player.stash_score())
                        self.avg_reward_per_action.append(- player.stash_score())
            if debug:
                print(f'Rewards gained in this turn: {self.rewards_per_game[-1]}')

            for player in self.rummy.players:
                player.points = player.stash_score()
                if player.name == 'Yike':
                    self.scores_end_game.append(player.points)
                if debug:
                    print(f'{player.name}: {player.points}')

        print(f'Per Group Training Time: {(time.time() - start_time) / 60:.1f} minutes')


# %%
agent = RLAgent(env=RummyAgent([Player('Yike', list(), isBot=False),  # not bot - train and use policy
                                Player('Bot 1', list(), isBot=True),
                                Player('Bot 2', list(), isBot=True),
                                Player('Bot 3', list(), isBot=True)], max_card_length=3, max_turns=20))
agent.train(gamma=0.9,
            alpha=0.2,  # learning rate
            epsilon=0.1,  # probability to choose random action
            maxiter=10,
            reset=True,
            debug=True)

agent.Q_pick.shape
agent.Q_drop.shape
agent.scores_end_game

# %% defines the function to run agents with different hyparameter settings and records their results in terms of end score after 100 batches of 10000 games.


def grid_search_RL_params(gamma_input: list, alpha_input: list, epsilon_input: list, debug=False):
    """
    Parameters
    ----------
    gamma_input: a list of gammas, discount rate
    alpha_input: a list of alpha, learning rate
    epsilon_input: a list of epsilon, the greedy index

    Returns
    -------
    result_df: df which stores the param combinations and the performance
    """
    columns = ['RL_param', 'Average End Score', 'Negative Reward per Action']
    result_df = pd.DataFrame(columns=columns)  # create empty dataframe to store results
    import itertools
    gs_options = list(itertools.product(gamma_input, alpha_input, epsilon_input))
    for param_combination in gs_options:
        (gamma, alpha, epsilon) = param_combination
        print(f'Hyperparameter {param_combination} training starts...')
        # instantiate and reset
        agent = RLAgent(env=RummyAgent([Player('Yike', list(), isBot=False),  # not bot - train and use policy
                                        Player('Bot 1', list(), isBot=True),
                                        Player('Bot 2', list(), isBot=True),
                                        Player('Bot 3', list(), isBot=True)], max_card_length=3, max_turns=20))
        for _ in range(3):
            agent.train(gamma=gamma, alpha=alpha, epsilon=epsilon, maxiter=int(1e5), reset=False, debug=False)
            # print(len(agent.scores_end_game))

        #         print(f'Hyperparameter {param_combination} training complete.')
        avg_end_score = np.mean(agent.scores_end_game[-int(5e4):])  # take the last 100000 games
        neg_r_per_action = -np.mean(agent.avg_reward_per_action[-int(5e4):])  # take the last 100000 games
        result_df = result_df.append(
            {'RL_param': param_combination, 'Average End Score': avg_end_score,
             'Negative Reward per Action': neg_r_per_action}, ignore_index=True)
    if debug:
        return agent.scores_end_game, agent.avg_reward_per_action, result_df
    else:
        return result_df


gs_result = grid_search_RL_params([0.9, 0.95, 0.99], [0.1, 0.2], [0.1, 0.2])
# end_score, r_per_action, gs_result = grid_search_RL_params([0.9, 0.99], [0.1], [0.1], debug=True)
print(gs_result)  # prints the hyperparameter settings and corresponding post training end score

# plot
import warnings

warnings.filterwarnings('ignore')  # suppress warning on legend

fig, ax1 = plt.subplots(figsize=(14, 7), dpi=100)
gs_result.plot(x='RL_param', y='Average End Score', marker='o', ax=ax1, color='C0', label='_nolegend_')
ax1.set_xlabel('RL parameters\n(gamma, alpha, epsilon)', fontsize=12)
ax1.set_ylabel('Average End Score', color='C0', fontsize=15)
ax1.tick_params(axis='y', labelcolor='C0')
ax1.set_xticks(np.arange(3 * 2 * 2))
ax1.set_xticklabels(gs_result.RL_param, rotation=20)

ax2 = ax1.twinx()
gs_result.plot(x='RL_param', y='Negative Reward per Action', marker='o', ax=ax2, color='C3', label='_nolegend_')
ax2.set_ylabel('Negative Reward per Action', color='C3', fontsize=15)
ax2.tick_params(axis='y', labelcolor='C3')

plt.axvline(x=2, color='gray', linestyle=':', linewidth=2, label='Selected Params')
plt.title('Average End Score and Average Negative Reward per Action under each Hyperparameter Combination', fontsize=12)
plt.legend()
plt.show()

# %% trains a new agent with the best hyperparameter setting in the above selection process and records its end score and average rewards after each training batch
avg_end_score = []
r_per_action = []
r_per_game = []
agent = RLAgent(env=RummyAgent([Player('Yike', list(), isBot=False),  # not bot - train and use policy
                                Player('Bot 1', list(), isBot=True),
                                Player('Bot 2', list(), isBot=True),
                                Player('Bot 3', list(), isBot=True)], max_card_length=3, max_turns=20))
for _ in range(10):
    print(f'Training Group {_ + 1}:')
    agent.train(gamma=0.9, alpha=0.2, epsilon=0.1, maxiter=5000, reset=False, debug=False)
    avg_end_score.append(np.mean(agent.scores_end_game))
    r_per_action.append(np.mean(agent.avg_reward_per_action))
    r_per_game.append(np.mean(agent.rewards_per_game))
    print(
        f'Average Endgame Stash Score: {avg_end_score[-1]}\n'
        f'Average Reward per Action Taken: {r_per_action[-1]}\n'
        f'Average Reward per Game: {r_per_game[-1]}\n'
          )

    if np.isnan(r_per_action[-1]):
        print(r_per_action)
        break

plt.subplots(figsize=(8, 4), dpi=100)
plt.plot(avg_end_score, marker='o', color='b', markersize=3)
plt.ylabel('Average Endgame Stash Score')
plt.xlabel('# of Training Groups')
plt.xticks(np.arange(0, 85, 5))
plt.title('Average Endgame Stash Score as Training Groups Grow')
plt.show()

plt.subplots(figsize=(8, 4), dpi=120)
plt.plot(r_per_action, marker='o', color='C3', markersize=3)
plt.ylabel('Average Reward per Action')
plt.xlabel('# of Training Groups')
plt.xticks(np.arange(0, 85, 5))
plt.title('Average Reward per Action as Training Groups Grow')
plt.show()

plt.subplots(figsize=(8, 4), dpi=120)
plt.plot(r_per_game, marker='o', color='C2', markersize=3)
plt.ylabel('Average Reward per Game')
plt.xlabel('# of Training Groups')
plt.xticks(np.arange(0, 85, 5))
plt.title('Average Reward per Game as Training Groups Grow')
plt.show()
