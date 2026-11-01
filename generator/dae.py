"""Collada (.dae) writer for BeamNG flexbody meshes.

Each Mesh becomes one geometry plus one node in the visual scene; the node name
is what a jbeam "flexbodies" row refers to. Coordinates are written as-is in
vehicle space (Z up, metres). Two UV sets are written (set 0 = texture UVs,
set 1 = a copy) because some BeamNG material layers read the second set.
"""
import numpy as np
from xml.sax.saxutils import escape


def _fmt(a, digits=5):
    a = np.asarray(a, float).ravel()
    return " ".join(f"{x:.{digits}f}".rstrip("0").rstrip(".") if x != 0 else "0" for x in a)


def _ids(name):
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return safe


def write_dae(path, meshes):
    names = [m.name for m in meshes]
    if len(set(names)) != len(names):
        raise ValueError("duplicate mesh names: %s" % sorted({n for n in names if names.count(n) > 1}))
    for n in names:
        if n[-1].isdigit():
            # Torque's shape loader reads trailing digits as a level-of-detail size.
            raise ValueError(f"mesh name must not end in a digit: {n}")
    materials = sorted({mat for m in meshes for mat in m.groups})
    out = []
    w = out.append
    w('<?xml version="1.0" encoding="utf-8"?>')
    w('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">')
    w('  <asset><contributor><authoring_tool>robotaxi generator</authoring_tool></contributor>'
      '<unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>')
    w('  <library_effects>')
    for mat in materials:
        mid = _ids(mat)
        w(f'    <effect id="{mid}-effect"><profile_COMMON><technique sid="common"><lambert>'
          f'<diffuse><color sid="diffuse">0.8 0.8 0.8 1</color></diffuse></lambert></technique></profile_COMMON></effect>')
    w('  </library_effects>')
    w('  <library_materials>')
    for mat in materials:
        mid = _ids(mat)
        w(f'    <material id="{mid}" name="{escape(mat)}"><instance_effect url="#{mid}-effect"/></material>')
    w('  </library_materials>')
    w('  <library_geometries>')
    for m in meshes:
        gid = _ids(m.name) + "-mesh"
        nv = len(m.v)
        w(f'    <geometry id="{gid}" name="{escape(m.name)}"><mesh>')
        w(f'      <source id="{gid}-positions"><float_array id="{gid}-positions-array" count="{nv*3}">{_fmt(m.v)}</float_array>'
          f'<technique_common><accessor source="#{gid}-positions-array" count="{nv}" stride="3">'
          f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>')
        w(f'      <source id="{gid}-normals"><float_array id="{gid}-normals-array" count="{nv*3}">{_fmt(m.n, 4)}</float_array>'
          f'<technique_common><accessor source="#{gid}-normals-array" count="{nv}" stride="3">'
          f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>')
        for s in (0, 1):
            w(f'      <source id="{gid}-map{s}"><float_array id="{gid}-map{s}-array" count="{nv*2}">{_fmt(m.uv, 5)}</float_array>'
              f'<technique_common><accessor source="#{gid}-map{s}-array" count="{nv}" stride="2">'
              f'<param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>')
        w(f'      <vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/></vertices>')
        for mat, f in m.groups.items():
            if len(f) == 0:
                continue
            mid = _ids(mat)
            idx = np.repeat(np.asarray(f, int).ravel()[:, None], 4, axis=1).ravel()
            w(f'      <triangles material="{mid}" count="{len(f)}">'
              f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
              f'<input semantic="NORMAL" source="#{gid}-normals" offset="1"/>'
              f'<input semantic="TEXCOORD" source="#{gid}-map0" offset="2" set="0"/>'
              f'<input semantic="TEXCOORD" source="#{gid}-map1" offset="3" set="1"/>'
              f'<p>{" ".join(map(str, idx))}</p></triangles>')
        w('    </mesh></geometry>')
    w('  </library_geometries>')
    w('  <library_visual_scenes><visual_scene id="Scene" name="Scene">')
    for m in meshes:
        gid = _ids(m.name) + "-mesh"
        w(f'    <node id="{_ids(m.name)}" name="{escape(m.name)}" type="NODE">'
          f'<matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
          f'<instance_geometry url="#{gid}" name="{escape(m.name)}"><bind_material><technique_common>')
        for mat in m.groups:
            mid = _ids(mat)
            w(f'      <instance_material symbol="{mid}" target="#{mid}">'
              f'<bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/></instance_material>')
        w('    </technique_common></bind_material></instance_geometry></node>')
    w('  </visual_scene></library_visual_scenes>')
    w('  <scene><instance_visual_scene url="#Scene"/></scene>')
    w('</COLLADA>')
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
