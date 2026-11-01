"""Attempt 4 (2026-11-01): the pipeline of the 5-output network (README "Attempt 4: PRE-REGISTRATION ADDENDUM").

Network outputs (E, 5): speed share; junction-approach share; three intent logits (stay, pass left, pass right).
  * speed: min(share x 1.1 x limit, limit, 25), then the flat parked-car caution (caution.py);
  * junction approach: from 60 m before the next stop line or junction until the car is through its box, also at most
    (0.3 + 0.7 sigmoid(junction output)) x limit: how fast to approach a junction is the network's decision;
  * intent: argmax; stay asks for offset 0, pass left for -0.5 m, pass right for +0.5 m (the shield widens a pass
    request to the offset that clears the group it passes, or refuses it);
  * guards: brains2.guards2 with the yellow at 3.5 m/s2, the shipped network's rule (Drive keeps 2.2);
  * then exactly the attempt-3 machinery of brains2.nn_actions2: the commitment, the in-lane check, the shield.
"""
import numpy as np

import brains2
import caution
import shield
import sim
from brains import V_MAX
from brains2 import RowView, guards2

YELLOW = 3.5
J_NEAR, J_BOX = 60.0, 11.0          # junction zone: 60 m before the stop line until the front is 11 m past it (the box)
INTENT_OFFSET = 0.5
LAST = {"lateral": 0, "mode": np.zeros(8, dtype=int)}


def junction_zone(w):
    """(E,) the car is within 60 m before a stop line or junction, or inside its box"""
    sc = w.sc
    dist = sc.jS - 6.0 - (w.es[:, None] + 2.6)
    return ((sc.jKind != sim.J_NONE) & (dist < J_NEAR) & (dist > -J_BOX)).any(axis=1)


def nn_actions4(out, w, use_shield=True):
    E = w.E
    vlim = w.vlim_at(w.es)
    share = 1 / (1 + np.exp(-out[:, 0]))
    vT_net = np.minimum(np.minimum(share * 1.1 * vlim, vlim), V_MAX)
    vT_net = np.minimum(vT_net, caution.cap(w, "flat"))
    jshare = 0.3 + 0.7 / (1 + np.exp(-out[:, 1]))
    vT_net = np.where(junction_zone(w), np.minimum(vT_net, jshare * vlim), vT_net)
    intent = np.argmax(out[:, 2:5], axis=1)
    sc = w.sc
    lo = sc.edgeL + sim.EGO_W / 2 + 0.2
    hi = sc.edgeR - sim.EGO_W / 2 - 0.2
    dT_c = np.clip(np.where(intent == 1, -INTENT_OFFSET, np.where(intent == 2, INTENT_OFFSET, 0.0)), lo, hi)
    commit = getattr(w, "a4_commit", None)
    if commit is None:
        commit = w.a4_commit = np.zeros(E)
    keep = (commit != 0) & (np.sign(dT_c) == np.sign(commit)) & (np.abs(dT_c) < np.abs(commit)) & (np.abs(dT_c) > shield.OUT_LANE)
    dT_c = np.where(keep, commit, dT_c)
    v_lane, d_lane = guards2(w, with_stop=True, yellow_decel=YELLOW)
    vT = np.minimum(vT_net, v_lane)
    lat_req = (np.abs(dT_c - w.ed) > shield.OUT_LANE) | (np.abs(w.ed) > shield.OUT_LANE) | (np.abs(dT_c) > shield.OUT_LANE)
    lat_req &= ~w.done
    dT = np.clip(dT_c, -shield.OUT_LANE, shield.OUT_LANE)
    if not use_shield:
        # ablation: the request goes straight to the controller (a pass at the request's 0.5 m), the guards still cap
        return vT, dT_c
    rows = np.where(lat_req)[0]
    modes = np.zeros(8, dtype=int)
    keep_rows = np.where(~lat_req & ~w.done)[0]
    vI, mI = shield.shield_inlane(w, vT, d_lane, keep_rows)
    vT[keep_rows] = vI
    modes[6] = int((mI == 6).sum())
    modes[7] = int((mI == 7).sum())
    if len(rows):
        def cap_fn(r, offs):
            return guards2(RowView(w, r), off=np.asarray(offs, dtype=float), update_latch=False, lead_onc=False,
                           with_stop=True, yellow_decel=YELLOW)
        vS, dS, mode = shield.shield(w, vT_net, dT_c, rows, cap_fn)
        vT[rows], dT[rows] = vS, dS
        commit[rows] = np.where((mode == 0) | (mode == 2), dS, 0.0)
        for m_ in range(6):
            modes[m_] = int((mode == m_).sum())
    commit[~lat_req] = 0.0
    LAST["lateral"] = len(rows)
    LAST["mode"] = modes
    return vT, dT


class Net5:
    """91 -> 64 -> 64 -> 5, tanh hidden layers; parameter vector [W1, b1, W2, b2, W3, b3]. The first layer is frozen in
    attempt 4; TRAIN is the slice of the vector that CMA-ES moves (W2, b2, W3, b3)."""
    shapes = [(sim.OBS_DIM, 64), (64,), (64, 64), (64,), (64, 5), (5,)]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)
    TRAIN = slice(sizes[0] + sizes[1], n)

    @classmethod
    def unpack(cls, theta):
        out, i = [], 0
        for sh, sz in zip(cls.shapes, cls.sizes):
            out.append(theta[..., i:i + sz].reshape(theta.shape[:-1] + sh))
            i += sz
        return out

    @classmethod
    def forward(cls, thetas, obs):
        """thetas (P, n), obs (P, M, 91) -> (P, M, 5); the first layer is shared when all members have the same one"""
        W1, b1, W2, b2, W3, b3 = cls.unpack(thetas)
        h1 = np.tanh(np.matmul(obs, W1) + b1[:, None, :])
        h2 = np.tanh(np.matmul(h1, W2) + b2[:, None, :])
        return np.matmul(h2, W3) + b3[:, None, :]

    @classmethod
    def from_a2(cls, theta2):
        """the warm start (pre-registered): hidden layers and speed head of the attempt-2 network; junction head with no
        effect; intent logits (0, -10 o2 - 1.13, 10 o2 - 1.13) from its offset output o2 (its pass decisions)"""
        import brains
        W1, b1, W2, b2, W3o, b3o = [a[0] for a in brains.MLP().unpack(theta2[None, :])]
        W3 = np.zeros((64, 5))
        b3 = np.zeros(5)
        W3[:, 0], b3[0] = W3o[:, 0], b3o[0]
        W3[:, 1], b3[1] = 0.0, 3.0
        W3[:, 3], b3[3] = -10 * W3o[:, 1], -10 * b3o[1] - 1.13
        W3[:, 4], b3[4] = 10 * W3o[:, 1], 10 * b3o[1] - 1.13
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
