from typing import Optional, Dict
import os
import re
import string


def _build_format_regex(format_str: str) -> re.Pattern:
    """Build a regex that parses filenames produced by `format_str.format(**data)`
    back into their named fields, so an existing checkpoint dir can be reconciled
    on resume instead of starting from an empty path_value_map.
    """
    pattern = ['^']
    for literal_text, field_name, format_spec, _ in string.Formatter().parse(format_str):
        pattern.append(re.escape(literal_text))
        if field_name is None:
            continue
        spec = format_spec or ''
        if spec.endswith(('d', 'x', 'X', 'o', 'b')):
            group = r'-?\d+'
        elif spec.endswith(('f', 'e', 'E', 'g', 'G', '%')) or spec.startswith('.'):
            group = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        else:
            group = r'.+?'
        pattern.append(f'(?P<{field_name}>{group})')
    pattern.append('$')
    return re.compile(''.join(pattern))


def _scan_existing_checkpoints(save_dir: str, format_str: str, monitor_key: str) -> Dict[str, float]:
    """Reconcile an existing checkpoints/ dir (e.g. after training.resume) with the
    value this manager sorts by, so it doesn't think zero checkpoints exist and
    let old ones pile up forever instead of being rotated out.
    """
    path_value_map = {}
    if not os.path.isdir(save_dir):
        return path_value_map

    regex = _build_format_regex(format_str)
    for filename in os.listdir(save_dir):
        match = regex.match(filename)
        if match is None or monitor_key not in match.groupdict():
            continue
        try:
            value = float(match.group(monitor_key))
        except ValueError:
            continue
        path_value_map[os.path.join(save_dir, filename)] = value
    return path_value_map


class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = _scan_existing_checkpoints(save_dir, format_str, monitor_key)

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path

class LastNCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            k: int,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.k = k
        self.format_str = format_str
        self.path_value_map = _scan_existing_checkpoints(save_dir, format_str, monitor_key)

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if value > min_value:
            delete_path = min_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path

# class CheckpointManager:
#     def __init__(self,
#             save_dir,
#             monitor_key: str,
#             format_str='epoch={epoch:03d}.ckpt'
#         ):

#         self.save_dir = save_dir
#         self.monitor_key = monitor_key
#         self.format_str = format_str
#         self.path_value_map = dict()
    
#     def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:

#         value = data[self.monitor_key]
#         ckpt_path = os.path.join(
#             self.save_dir, self.format_str.format(**data))
#         self.path_value_map[ckpt_path] = value
#         return ckpt_path