#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
import random
import enum
import time


class GoBangResult(enum.IntEnum):
    BLACK = 1
    WHITE = 2
    TIE = 3
    UNKNOWN = 4


class GoBangPlayer(enum.IntEnum):
    BLACK = 1
    WHITE = 2


class PointState(enum.IntEnum):
    BLACK = 1
    WHITE = 2
    EMPTY = 3


class GoBang:

    def __init__(self, cell_num: int):
        self._cell_num = cell_num
        self._board = [[PointState.EMPTY] * cell_num for _ in range(cell_num)]
        self._steps = 0
        self._result = GoBangResult.UNKNOWN

    def board(self):
        return self._board

    def cell_num(self):
        return self._cell_num

    def result(self):
        return self._result

    def player(self):
        if self._steps % 2 == 0:
            return GoBangPlayer.BLACK
        else:
            return GoBangPlayer.WHITE

    def terminated(self):
        return self._result != GoBangResult.UNKNOWN

    def evaluate(self, i, j):

        def equal(x, y):
            if not 0 <= x < self._cell_num:
                return False
            if not 0 <= y < self._cell_num:
                return False
            if not self._board[x][y] == self._board[i][j]:
                return False
            return True

        if self._steps == self._cell_num * self._cell_num:
            return GoBangResult.TIE
        directs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direct in directs:
            left = 1
            while left < 5:
                i2 = i - direct[0] * left
                j2 = j - direct[1] * left
                if equal(i2, j2):
                    left += 1
                    continue
                else:
                    break
            right = 1
            while right < 5:
                i2 = i + direct[0] * right
                j2 = j + direct[1] * right
                if equal(i2, j2):
                    right += 1
                    continue
                else:
                    break
            if left + right >= 6:
                if self._board[i][j] == PointState.BLACK:
                    return GoBangResult.BLACK
                if self._board[i][j] == PointState.WHITE:
                    return GoBangResult.WHITE
        return GoBangResult.UNKNOWN

    def step(self, i: int, j: int):
        if self._result != GoBangResult.UNKNOWN:
            return
        if 0 <= i < self._cell_num and 0 <= j < self._cell_num:
            print('i', i, 'j', j, self._board)
            if self._board[i][j] != PointState.EMPTY:
                print("已经落子过")
            elif self._steps % 2 == 0:
                self._board[i][j] = PointState.BLACK
                self._steps += 1
            else:
                self._board[i][j] = PointState.WHITE
                self._steps += 1
        self._result = self.evaluate(i, j)


class Strategy:

    def get_action(self, board, player):
        raise NotImplementedError


class RandomStrategy:

    def get_action(self, board, player):
        cell_num = len(board)
        while True:
            i = random.randint(0, cell_num - 1)
            j = random.randint(0, cell_num - 1)
            if board[i][j] != PointState.EMPTY:
                continue
            return i, j


class Render:

    def __init__(self, gobang: GoBang, space=60, cell_size=60,
                 black_strategy: Strategy = None,
                 white_strategy: Strategy = None):
        self._white_strategy = white_strategy
        self._black_strategy = black_strategy
        self._gobang = gobang
        self._space = space  # 四周留下的边距
        self._cell_size = cell_size  # 每个格子大小
        self._cell_num = gobang.cell_num()
        self._cycle_size = int(.4 * self._cell_size)
        self._grid_size = self._cell_size * (self._cell_num - 1) + self._space * 2  # 棋盘的大小
        self._position = [0, 0]
        pygame.display.set_caption('Easy Five Game')
        pygame.font.init()
        self._font = pygame.font.SysFont("Times New Roman", 48)
        self._screen = pygame.display.set_mode((self._grid_size, self._grid_size))  # 设置窗口长宽

    def start(self):
        while True:
            player = self._gobang.player()
            if not self._gobang.terminated() and player == GoBangPlayer.BLACK and self._black_strategy is not None:
                i, j = self._black_strategy.get_action(self._gobang.board(), player)
                self._gobang.step(i, j)
            if not self._gobang.terminated() and player == GoBangPlayer.WHITE and self._white_strategy is not None:
                i, j = self._white_strategy.get_action(self._gobang.board(), player)
                self._gobang.step(i, j)
            for event in pygame.event.get():
                self.handle_event(event)
            self.render()
            time.sleep(1)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONUP:  # 鼠标弹起
            x, y = pygame.mouse.get_pos()  # 获取鼠标位置
            x = int(round((x - self._space) * 1.0 / self._cell_size))  # 获取到x方向上取整的序号
            y = int(round((y - self._space) * 1.0 / self._cell_size))  # 获取到y方向上取整的序号
            if x == self._cell_num and y == -1:
                return
            if 0 <= x < self._cell_num and 0 <= y < self._cell_num:
                self._position = [x, y]
                self._gobang.step(x, y)
        if event.type == pygame.KEYDOWN:
            print('key', event.key)
            if event.key == pygame.K_LEFT:
                self._position[0] = max(0, self._position[0] - 1)
            elif event.key == pygame.K_RIGHT:
                self._position[0] = min(self._cell_num - 1, self._position[0] + 1)
            if event.key == pygame.K_UP:
                self._position[1] = max(0, self._position[1] - 1)
            elif event.key == pygame.K_DOWN:
                self._position[1] = min(self._cell_num - 1, self._position[1] + 1)
            elif event.key == pygame.K_RETURN:
                self._gobang.step(*self._position)

    def render(self):
        self._screen.fill((50, 100, 50))  # 将界面设置为蓝色
        for x in range(0, self._cell_size * self._cell_num, self._cell_size):
            pygame.draw.line(self._screen, (200, 200, 200), (x + self._space, 0 + self._space),
                             (x + self._space, self._cell_size * (self._cell_num - 1) + self._space), 1)
        for y in range(0, self._cell_size * self._cell_num, self._cell_size):
            pygame.draw.line(self._screen, (200, 200, 200), (0 + self._space, y + self._space),
                             (self._cell_size * (self._cell_num - 1) + self._space, y + self._space), 1)
        for i in range(self._cell_num):
            for j in range(self._cell_num):
                board = self._gobang.board()
                center = (i * self._cell_size + self._space, j * self._cell_size + self._space)
                if board[i][j] == PointState.BLACK:
                    pygame.draw.circle(self._screen,
                                       (0, 0, 0),
                                       center,
                                       self._cycle_size)
                elif board[i][j] == PointState.WHITE:
                    pygame.draw.circle(self._screen,
                                       (255, 255, 255),
                                       center,
                                       self._cycle_size)
        i, j = self._position
        position = (
                int((i - 0.5) * self._cell_size + self._space),
                int((j - 0.5) * self._cell_size + self._space),
                self._cell_size,
                self._cell_size,
                )
        pygame.draw.rect(self._screen, (255, 255, 255), position, 1)
        img = self._font.render("restart", True, (15, 65, 45))
        self._screen.blit(img, (self._cell_num * self._cell_size, 20))
        result = self._gobang.result()
        out_msg_dict = {
            GoBangResult.WHITE: "White wins",
            GoBangResult.BLACK: "Black wins",
            GoBangResult.TIE: "Ties"
        }
        if result in out_msg_dict:
            img = self._font.render(out_msg_dict[result], True, (25, 25, 25))
            self._screen.blit(img, (20, 20))
        pygame.display.update()  # 必须调用update才能看到绘图显示


if __name__ == "__main__":
    gobang = GoBang(7)
    render = Render(gobang,
                    black_strategy=RandomStrategy(),
                    white_strategy=RandomStrategy())
    render.start()
