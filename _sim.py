from expert_core import *
import numpy as np
np.set_printoptions(linewidth=200)

for rule in ("always", "on_error"):
    for norm in (False, True):
        m = ExpertModel(update_rule=rule, normalize=norm)
        rep = m.train()
        print(rule, "norm=%s" % norm, "epochs", rep.n_epochs, "conv", rep.converged,
              "during", rep.curve_during, "after", rep.curve_after)

m = ExpertModel()
rep = m.train(max_epochs=4, auto_stop=False)
print(m.matrix)
print("\n".join(m.equations()))
print(m.check_all()[0])
r = m.explain([1, 1, 0, 0, 0, 0, 0, 1], "Скрипка")
print(r["text"])
print("----- unknown")
for name, v in UNKNOWN_OBJECTS.items():
    r = m.expertise(v, name)
    print(name, "->", r["winner_name"], fmt(r["gap"]))
print("----- step study")
for row in step_study(m):
    print(row["step"], row["epochs"], row["converged"], row["curve_after"])
print("----- feedback")
rep = m.learn_feedback([1, 0, 1, 0, 1, 0, 0, 1], "Клавесин")
print(rep.kind, rep.n_epochs, rep.converged, rep.added_example, rep.curve_after)
print(m.expertise([1, 0, 1, 0, 1, 0, 0, 1])["winner_name"])
print(m.problems())
print(m.world_distances()[("Скрипка", "Гитара")], m.mind_distances()[("Скрипка", "Гитара")])
m.add_feature("Медный"); m.add_class("Труба"); m.add_example("Труба", [0,0,0,1,0,0,0,0,1])
rep = m.train(); print(rep.n_epochs, rep.converged, m.matrix.shape)
m.save_json("/tmp/base.json"); m2 = ExpertModel.load_json("/tmp/base.json"); print(m2.n_features, m2.n_classes, m2.trained)
