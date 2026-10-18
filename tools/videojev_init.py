#!/usr/bin/env python3
"""Дообучение С ВЕСОВ VideoJev-4B, а не с чистой Qwen3.5-4B.

Официальный рецепт VideoJev учит модель с базовой Qwen3.5-4B. На 39 видео это дало бы модель
хуже готовой VideoJev-4B: всё, чему её научили на больших датасетах, пропало бы. Поэтому:

  lm    собирает папку в формате Qwen3.5-4B (config, токенизатор), но с весами языковой и визуальной
        части из VideoJev-4B. Ключи сопоставляются по окончанию имени и форме тензора, с проверкой.
  head  берёт head.pt из этапа head_only как шаблон (имена и формы) и заполняет его весами головы
        из VideoJev-4B. Результат передаётся в этап joint через --init-head.

  python tools/videojev_init.py lm   --qwen Q --videojev V --out OUT
  python tools/videojev_init.py head --template head.pt --videojev V --out head_from_videojev.pt
"""
import argparse
import json
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path


# ---------- сопоставление ключей (чистый Python, без torch) ----------

def match_one(key, shape, pool, used):
    """Ключ из pool с тем же окончанием имени и той же формой.

    Окончание укорачивается слева, пока не найдётся ровно один кандидат.
    Если на каком-то уровне кандидатов несколько, совпадение считается неоднозначным.
    """
    parts = key.split(".")
    for i in range(len(parts)):
        suf = ".".join(parts[i:])
        cands = [k for k, s in pool.items()
                 if k not in used and s == shape and (k == suf or k.endswith("." + suf))]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return None
    return None


def match_keys(src, pool):
    """src, pool: {имя: форма}. Возвращает (mapping src->pool, список ненайденных)."""
    used, mapping, missing = set(), {}, []
    for k, shape in src.items():
        m = match_one(k, shape, pool, used)
        if m is None:
            missing.append(k)
        else:
            mapping[k] = m
            used.add(m)
    return mapping, missing


# ---------- работа с файлами ----------

def index_safetensors(model_dir):
    from safetensors import safe_open
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                out[k] = (f, tuple(sf.get_slice(k).get_shape()))
    if not out:
        sys.exit(f"Нет *.safetensors в {model_dir}")
    return out


def cmd_lm(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    q, v = index_safetensors(a.qwen), index_safetensors(a.videojev)
    mapping, missing = match_keys({k: s for k, (_, s) in q.items()}, {k: s for k, (_, s) in v.items()})
    print(f"Тензоров в Qwen3.5-4B: {len(q)}; найдено в VideoJev-4B: {len(mapping)}; не найдено: {len(missing)}")
    expected_missing = {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.31.mlp.down_proj.weight",
        "model.language_model.layers.31.mlp.gate_proj.weight",
        "model.language_model.layers.31.mlp.up_proj.weight",
        "model.language_model.layers.31.post_attention_layernorm.weight",
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    if set(missing) != expected_missing:
        for k in missing[:20]:
            print("  не найдено:", k, q[k][1])
        print("  примеры ключей VideoJev:", list(v)[:10])
        sys.exit("Непредусмотренные несовпадения весов. Пришлите этот вывод.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(a.qwen).iterdir():  # config, токенизатор, процессор
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)

    weight_map, shard, shard_bytes, n_shard, total, changed = {}, {}, 0, 0, 0, 0

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if shard:
            n_shard += 1
            name = f"model-{n_shard:05d}.safetensors"
            save_file(shard, str(out / name), metadata={"format": "pt"})
            for k in shard:
                weight_map[k] = name
            shard, shard_bytes = {}, 0

    with ExitStack() as st:
        handles = {}

        def get(path, key):
            if path not in handles:
                handles[path] = st.enter_context(safe_open(str(path), "pt"))
            return handles[path].get_tensor(key)

        for k, (qf, _) in q.items():
            base = get(qf, k)
            if k in mapping:
                t = get(v[mapping[k]][0], mapping[k]).to(base.dtype)
                changed += int(not torch.equal(t, base))
            else:
                t = base
            shard[k] = t.contiguous()
            shard_bytes += t.numel() * t.element_size()
            total += t.numel() * t.element_size()
            if shard_bytes > 4e9:
                flush()
        flush()

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    print(f"Тензоров, отличающихся от Qwen3.5-4B (дообученных в VideoJev): {changed}")
    if changed == 0:
        print("[!] Ни один тензор не отличается от Qwen3.5-4B: проверьте пути к моделям.")
    print(f"Готово: {out}")


def find_state_dict(obj, path=()):
    """Словарь тензоров внутри head.pt (сам объект или вложенный state_dict)."""
    import torch
    if isinstance(obj, dict) and obj and all(isinstance(x, torch.Tensor) for x in obj.values()):
        return obj, path
    if isinstance(obj, dict):
        for key in ("state_dict", "head", "model", "module"):
            if key in obj:
                found = find_state_dict(obj[key], path + (key,))
                if found[0] is not None:
                    return found
        for key, val in obj.items():
            found = find_state_dict(val, path + (key,))
            if found[0] is not None:
                return found
    return None, path


def cmd_head(a):
    import torch
    from safetensors import safe_open

    tpl = torch.load(a.template, map_location="cpu", weights_only=False)
    sd, path = find_state_dict(tpl)
    if sd is None:
        sys.exit(f"В {a.template} не найден словарь тензоров. Пришлите вывод: python -c \"import torch; print(torch.load('{a.template}', weights_only=False).keys())\"")
    v = index_safetensors(a.videojev)
    mapping, missing = match_keys({k: tuple(t.shape) for k, t in sd.items()}, {k: s for k, (_, s) in v.items()})
    print(f"Тензоров головы: {len(sd)}; найдено в VideoJev-4B: {len(mapping)}")
    if missing:
        for k in missing:
            print("  не найдено:", k, tuple(sd[k].shape))
        sys.exit("Голову VideoJev не удалось сопоставить. Пришлите этот вывод.")
    new = {}
    for k, t in sd.items():
        with safe_open(str(v[mapping[k]][0]), "pt") as sf:
            new[k] = sf.get_tensor(mapping[k]).to(t.dtype)
        print(f"  {k}  <-  {mapping[k]}")
    node = tpl
    for key in path[:-1]:
        node = node[key]
    if path:
        node[path[-1]] = new
    else:
        tpl = new
    torch.save(tpl, a.out)
    print(f"Готово: {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lm")
    p.add_argument("--qwen", required=True)
    p.add_argument("--videojev", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("head")
    p.add_argument("--template", required=True)
    p.add_argument("--videojev", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"lm": cmd_lm, "head": cmd_head}[a.cmd](a)


if __name__ == "__main__":
    main()
