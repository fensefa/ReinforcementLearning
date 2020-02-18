import tornado.ioloop
import tornado.web
import tornado.autoreload
import math
import numpy as np
from model import *

G_GO_SIZE = 10
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("template.html", go_size=G_GO_SIZE)

# g_model = RuleModel(G_GO_SIZE)
# g_model = DqnBaseModel(G_GO_SIZE)
g_model = CNNModel(G_GO_SIZE)
g_model.build_model()
g_model.restore()
class AIHandler(tornado.web.RequestHandler):
    def post(self):
        state = self.request.body.decode('utf-8')
        # print(state)
        np_state = np.zeros([G_GO_SIZE, G_GO_SIZE])
        for i in range(G_GO_SIZE):
            for j in range(G_GO_SIZE):
                # print(state, i, j, state[i*G_GO_SIZE+j])
                if state[i*G_GO_SIZE+j] == '0':
                    np_state[i][j] = 0
                elif state[i*G_GO_SIZE+j] == '-':
                    np_state[i][j] = -1
                elif state[i*G_GO_SIZE+j] == '+':
                    np_state[i][j] = 1
                else:
                    raise NotImplementedError
        result = g_model.result(np_state)
        print('='*20)
        print(np_state)
        if result[0] is not None:
            print(-1,-1,result)
            self.write("td-id-{}-{}:{}".format(0,0,result[0]))
            return
        i, j, score = g_model.predict(np_state)
        np_state[i][j] = 1
        result = g_model.result(np_state)
        print(i,j,result)
        self.write("td-id-{}-{}:{}".format(i+1,j+1,result[0]))

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/get_id", AIHandler),
        ],
        auto_reaload = True,
    )

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    app = make_app()
    app.listen(8088)
    tornado.autoreload.start()
    tornado.ioloop.IOLoop.current().start()
