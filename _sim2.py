from expert_core import *
import numpy as np
np.set_printoptions(linewidth=200)
m = ExpertModel()
rep = m.train()
print("default:", rep.n_epochs, rep.converged, rep.curve_during, rep.curve_after)
print(m.matrix)
for e in rep.epochs:
    print("epoch", e.number, [(p.class_name[:5], p.predicted_name[:5], p.ok, p.updated) for p in e.presentations])
print("----- unknown")
for name, v in UNKNOWN_OBJECTS.items():
    r = m.expertise(v, name); print(name, "->", r["winner_name"], fmt(r["gap"]), [fmt(d) for d in r["demons"]])
print("----- step study normalize on_error")
for row in step_study(m, steps=(0.05,0.1,0.2,0.25,0.5,1.0,2.0), max_epochs=100):
    print(row["step"], row["epochs"], row["converged"], row["curve_after"][:15])
print("----- step study no normalize")
for row in step_study(m, steps=(0.1,1.0,5.0), normalize=False, max_epochs=100):
    print(row["step"], row["epochs"], row["converged"], row["curve_after"][:15])
print("----- feedback")
r = m.expertise([1, 0, 1, 0, 1, 0, 0, 1]); print("before:", r["winner_name"])
rep = m.learn_feedback([1, 0, 1, 0, 1, 0, 0, 1], "Клавесин")
print(rep.kind, rep.n_epochs, rep.converged, rep.added_example, rep.curve_after)
print(m.expertise([1, 0, 1, 0, 1, 0, 0, 1])["winner_name"], m.errors_now())
print(m.problems())
m.add_feature("Медный"); m.add_class("Труба"); m.add_example("Труба", [0,0,0,1,0,0,0,0,1])
rep = m.train(); print(rep.n_epochs, rep.converged, m.matrix.shape, m.errors_now())
m.save_json("/tmp/base.json"); m2 = ExpertModel.load_json("/tmp/base.json"); print(m2.n_features, m2.n_classes, m2.trained)
m.remove_feature(8); m.remove_class(8); print(m.matrix.shape, len(m.examples))
print(m.describe_presentation(rep.epochs[0].presentations[1]))
print([ (f, fmt(v)) for f,v in m.feature_importance()])
