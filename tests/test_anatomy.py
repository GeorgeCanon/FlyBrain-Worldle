import numpy as np

from flybrain_worldle.connectome.anatomy import downsample_skeleton, parse_obj, prune_twigs, subsample_faces


def chain_with_branch():
    # 0-1-2-...-9 straight run, node 5 also has a side branch 10-11-12
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 10, 11])
    nodes = np.arange(len(parents), dtype=np.float32)[:, None].repeat(3, 1)
    return nodes, parents


def test_downsample_keeps_root_branch_and_leaves():
    nodes, parents = chain_with_branch()
    out_nodes, out_parents = downsample_skeleton(nodes, parents, every=3)
    kept = set(out_nodes[:, 0].astype(int).tolist())
    assert {0, 5, 9, 12} <= kept
    assert len(out_nodes) < len(nodes)
    assert out_parents[0] == -1
    assert (out_parents[1:] >= 0).all() and (out_parents < len(out_nodes)).all()
    # every kept node's parent is an ancestor in the original tree
    for i, p in enumerate(out_parents):
        if p < 0:
            continue
        a, target = int(out_nodes[i, 0]), int(out_nodes[p, 0])
        while a >= 0 and a != target:
            a = parents[a]
        assert a == target


def test_downsample_every_1_is_identity():
    nodes, parents = chain_with_branch()
    out_nodes, out_parents = downsample_skeleton(nodes, parents, every=1)
    assert len(out_nodes) == len(nodes)
    np.testing.assert_array_equal(out_parents, parents)


def test_prune_twigs_removes_short_terminal_branches_only():
    # trunk 0..10, a 1-node twig (11) off node 5, a 3-node branch (12-13-14) off node 7, a 2-node twig (15-16) off node 3
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 7, 12, 13, 3, 15])
    nodes = np.arange(len(parents), dtype=np.float32)[:, None].repeat(3, 1)
    out_nodes, out_parents = prune_twigs(nodes, parents, max_len=2)
    kept = set(out_nodes[:, 0].astype(int).tolist())
    assert not ({11, 15, 16} & kept)
    assert {12, 13, 14} <= kept and {0, 3, 5, 7, 10} <= kept
    assert out_parents[0] == -1 and (out_parents[1:] >= 0).all() and (out_parents < len(out_nodes)).all()


def test_parse_obj_and_subsample():
    text = "# comment\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2/1 4/1 3/1\n"
    v, f = parse_obj(text)
    assert v.shape == (4, 3) and f.shape == (2, 3)
    assert f.tolist() == [[0, 1, 2], [1, 3, 2]]
    v2, f2 = subsample_faces(v, f, max_faces=1)
    assert len(f2) == 1 and f2.max() < len(v2) and len(v2) == 3
