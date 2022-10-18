from collections import OrderedDict
from flax import linen as nn
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import bz2
import jax.numpy as jnp
import lark
import math
import os
import struct

class SpeciesEnergies(NamedTuple):
    species: jnp.ndarray
    energies: jnp.ndarray

def rebuild_model_ensemble(params):
    """Rebuilds ModuleEnsemble model network structure derived from params.

    Arguments:
        params (Dict): Dictionary of parameters from loaded NeuroChem or 
            restored checkpoint for a ModuleEnsemble model.

    Returns:
        An Ensemble of several ANIModel. Each ANImodel consists of several 
            atomic networks. An atomic network consists of 4 fully connected 
            layers with nn.celu activation function in between.
    """
    atomic_models = []
    atomic_ensemble_id = 0
    for atomic_params in params['params'].values():
        atomic_model = rebuild_atomic_ensemble(atomic_params, atomic_ensemble_id)
        atomic_models.append(atomic_model)
        atomic_ensemble_id += 1
    return ModuleEnsemble(atomic_models, name='ani')

def rebuild_atomic_ensemble(params, atomic_ensemble_id=0):
    """Rebuilds AtomicEnsemble model network structure derived from params.

    Arguments:
        params (Dict): Dictionary of parameters from loaded NeuroChem or 
            restored checkpoint for an AtomicEnsemble model.

    Returns:
        An ANIModel consisting of several atomic networks. An atomic network 
            consists of 4 fully connected layers with nn.celu activation
            function in between.
    """
    atomic_mlps = []
    atomic_mlp_id = 0
    for atomic_mlp_params in params.values():
        atomic_mlp = rebuild_atomic_mlp(atomic_mlp_params, atomic_mlp_id)
        atomic_mlps.append(atomic_mlp)
    return AtomicEnsemble(atomic_mlps, name=f'modules_{atomic_ensemble_id}')

def rebuild_atomic_mlp(params, atomic_mlp_id=0):
    """Rebuilds AtomicMLP model network structure derived from params. 
    
    By default the activation function of CELU with alpha=0.1 is hardcoded.

    Arguments:
        params (Dict): Dictionary of parameters from loaded NeuroChem or 
            restored checkpoint for an AtomicMLP model.

    Returns:
        An atomic network  consists of 4 fully connected layers with nn.celu 
            activation function in between.
    """
    features = []
    activation_fns = []
    for param_value in params.values():
      features.append(jnp.shape(param_value['bias'])[0])
      activation_fns.append(CELU(alpha=0.1))

    model = AtomicMLP(
        features=features, 
        activation_fns=activation_fns, 
        name=f'modules_{atomic_mlp_id}')
    return model

def jax_flatten(input, start_dim=0, end_dim=-1):
  if end_dim == -1:
    return input.reshape(-1)
  new_shape = []
  combined = 1
  # Flatten only starting from start_dim to end_dim
  for si, s in enumerate(input.shape):
    if si < start_dim or si > end_dim:
      new_shape.append(s)
    elif si >= start_dim and si < end_dim:
      combined *= s
    elif si == end_dim:
      new_shape.append(combined*s)
    else:
      raise ValueError("jax_flatten input shape is illegal")
  return input.reshape(tuple(new_shape))

def jax_index_select(input, index, axis):
  return jnp.take(input, index, axis)

def load_model_ensemble(species, prefix, count):
    """Returns an instance of :class:`jaxani.Ensemble` loaded from
    NeuroChem's network directories beginning with the given prefix.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        prefix (str): Prefix of paths of directory that networks configurations
            are stored.
        count (int): Number of models in the ensemble.

    Returns:
        An Ensemble of several ANIModel. Each ANImodel consists of several 
            atomic networks. An atomic network consists of 4 fully connected 
            layers with nn.celu activation function in between.
        A dictionary of parameters from loaded NeuroChem.
    """
    models = []
    params = {'params': {}}
    for ensemble_id in range(count):
        network_dir = os.path.join('{}{}'.format(prefix, ensemble_id), 'networks')
        model, param = load_model(species, network_dir, ensemble_id)
        models.append(model)
        params['params'].update({f'modules_{ensemble_id}': param['params']})
    return (ModuleEnsemble(models, name='ani'), params)

def load_model(species, dir_, ensemble_id=0):
    """Returns an instance of :class:`jaxani.AtomicEnsemble` loaded from
    NeuroChem's network directory.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        dir_ (str): String for directory storing network configurations.

    Returns:
        An ANIModel consisting of several atomic networks. An atomic network 
            consists of 4 fully connected layers with nn.celu activation
            function in between.
        A dictionary of parameters from loaded NeuroChem.
    """
    models = []
    params = {'params': {}}
    for species_id, species_name in enumerate(species):
        filename = os.path.join(dir_, 'ANN-{}.nnf'.format(species_name))
        model, param = load_atomic_network(filename, species_id)
        models.append(model)
        params['params'].update({f'modules_{species_id}': param['params']})
    return (AtomicEnsemble(models, name=f'modules_{ensemble_id}'), params)

def load_atomic_network(filename, species_id=0):
    """Returns an instance of :class:`AtomicModel` and a dictionary of 
        parameters from loaded NeuroChem's .nnf, .wparam and .bparam files."""

    networ_dir = os.path.dirname(filename)

    with open(filename, 'rb') as f:
        buffer_ = f.read()
        buffer_ = decompress_nnf(buffer_)
        layer_setups = parse_nnf(buffer_)

        features = []
        activation_fns = []
        params = {'params': {}}
        for i, layer_setup in enumerate(layer_setups):
            # construct linear layer and load parameters
            in_size = layer_setup['blocksize']
            out_size = layer_setup['nodes']
            wfn, wsz = layer_setup['weights']
            bfn, bsz = layer_setup['biases']
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            wfn = os.path.join(networ_dir, wfn)
            bfn = os.path.join(networ_dir, bfn)
            w, b = load_param_file(in_size, out_size, wfn, bfn)
            params['params'].update({f'fc{i+1}': {
                'kernel': jnp.array(w).reshape(out_size, in_size).T, 
                'bias': jnp.array(b).reshape(out_size)
                }})
            features.append(out_size)
            activation = _get_activation(layer_setup['activation'])
            if activation is not None:
                activation_fns.append(activation)
        
        model = AtomicMLP(
            features=features, 
            activation_fns=activation_fns, 
            name=f'modules_{species_id}')

        return (model, params)

def load_param_file(in_size, out_size, wfn, bfn):
    """Load `.wparam` and `.bparam` files"""
    wsize = in_size * out_size
    fw = open(wfn, 'rb')
    w = struct.unpack('{}f'.format(wsize), fw.read())
    fw.close()
    fb = open(bfn, 'rb')
    b = struct.unpack('{}f'.format(out_size), fb.read())
    fb.close()
    return w, b

def decompress_nnf(buffer_):
    while buffer_[0] != b'='[0]:
        buffer_ = buffer_[1:]
    buffer_ = buffer_[2:]
    return bz2.decompress(buffer_)[:-1].decode('ascii').strip()

def parse_nnf(nnf_file):
    # parse input file
    parser = lark.Lark(r'''
    identifier : CNAME

    inputsize : "inputsize" "=" INT ";"

    assign : identifier "=" value ";"

    layer : "layer" "[" assign * "]"

    atom_net : "atom_net" WORD "$" layer * "$"

    start: inputsize atom_net

    nans: "-"?"nan"

    value : SIGNED_INT
          | SIGNED_FLOAT
          | nans
          | "FILE" ":" FILENAME "[" INT "]"

    FILENAME : ("_"|"-"|"."|LETTER|DIGIT)+

    %import common.SIGNED_NUMBER
    %import common.LETTER
    %import common.WORD
    %import common.DIGIT
    %import common.INT
    %import common.SIGNED_INT
    %import common.SIGNED_FLOAT
    %import common.CNAME
    %import common.WS
    %ignore WS
    ''', parser='lalr')
    tree = parser.parse(nnf_file)

    # execute parse tree
    class TreeExec(lark.Transformer):

        def identifier(self, v):
            v = v[0].value
            return v

        def value(self, v):
            if len(v) == 1:
                v = v[0]
                if isinstance(v, lark.tree.Tree):
                    assert v.data == 'nans'
                    return math.nan
                assert isinstance(v, lark.lexer.Token)
                if v.type == 'FILENAME':
                    v = v.value
                elif v.type == 'SIGNED_INT' or v.type == 'INT':
                    v = int(v.value)
                elif v.type == 'SIGNED_FLOAT' or v.type == 'FLOAT':
                    v = float(v.value)
                else:
                    raise ValueError('unexpected type')
            elif len(v) == 2:
                v = self.value([v[0]]), self.value([v[1]])
            else:
                raise ValueError('length of value can only be 1 or 2')
            return v

        def assign(self, v):
            name = v[0]
            value = v[1]
            return name, value

        def layer(self, v):
            return dict(v)

        def atom_net(self, v):
            layers = v[1:]
            return layers

        def start(self, v):
            return v[1]

    layer_setups = TreeExec().transform(tree)
    return layer_setups

def _get_activation(activation_index):
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob/stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
    if activation_index == 6:
        return None
    elif activation_index == 5:  # Gaussian
        return Gaussian()
    elif activation_index == 9:  # CELU
        return CELU(alpha=0.1)
    else:
        raise NotImplementedError(
            'Unexpected activation {}'.format(activation_index))

class Gaussian():
    """Gaussian activation"""
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(- x * x)

class CELU():
    """CELU activation"""
    alpha: float

    def __init__(self, alpha: float = 1.) -> None:
        self.alpha = alpha

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        return nn.celu(input, self.alpha)

class Ensemble():
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        # super().__init__(modules)
        self.modules = modules
        self.size = len(modules)

    def __call__(self, species_input: Tuple[jnp.ndarray, jnp.ndarray],  # type: ignore
                cell: Optional[jnp.ndarray] = None,
                pbc: Optional[jnp.ndarray] = None) -> SpeciesEnergies:
        sum_ = 0
        for module in self.modules:
            sum_ += module(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)

def jax_masked_scatter(input, mask, source):
    return input.at[mask.nonzero()[0]].set(source)

class AtomicMLP(nn.Module):
    features: Sequence[int]
    activation_fns: Sequence[Any]

    def setup(self):
        self.layers = [nn.Dense(features=feat, name=f"fc{i+1}") for i, feat in enumerate(self.features)]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation_fns[i](x)
        return x

class AtomicEnsemble(nn.Module):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`jaxani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """
    
    modules: List[AtomicMLP]

    def setup(self):
      self.module_list = self.modules
    
    def __call__(self, species_aev: Tuple[jnp.ndarray, jnp.ndarray],  # type: ignore
                cell: Optional[jnp.ndarray] = None,
                pbc: Optional[jnp.ndarray] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, jnp.sum(atomic_energies, axis=1))

    def _atomic_energies(self, species_aev: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        # Obtain the atomic energies associated with a given ndarray of AEV's
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = jax_flatten(aev, 0, 1)

        output = jnp.zeros(species_.shape, dtype=aev.dtype)

        for i, m in enumerate(self.module_list):
            mask = (species_ == i)
            midx = mask.nonzero()[0].flatten()
            if midx.shape[0] > 0:
                input_ = jax_index_select(aev, midx, 0)
                output = jax_masked_scatter(output, mask, m(input_).flatten())
        output = output.reshape(species.shape)
        return output

class ModuleEnsemble(nn.Module):
    """Compute the average output of an ensemble of nn.Module's."""
    modules: Sequence[AtomicEnsemble]

    def setup(self):
        self.size = len(self.modules)

    def __call__(self, species_input: Tuple[jnp.ndarray, jnp.ndarray],  # type: ignore
                cell: Optional[jnp.ndarray] = None,
                pbc: Optional[jnp.ndarray] = None) -> SpeciesEnergies:
        sum_ = 0
        for module in self.modules:
            sum_ += module(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)