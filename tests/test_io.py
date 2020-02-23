import os
from collections import OrderedDict
import numpy as np
import h5py
import pytest
from neuroballad.io import IO, Input, Output
from pathlib import Path
dt = 1e-4
INPUT_DATA = OrderedDict(
    I=np.random.randn(1000, 4),
    g=np.random.randn(1000, 3)
)
INPUT_UIDS = OrderedDict(
    I=['0', '1', '2', '3'],
    g=['0', '1', '2']
)

OUTPUT_DATA = OrderedDict(
    V=np.random.randn(1000, 4),
    spike_state=np.random.randint(2, size=(1000, 3))
)
OUTPUT_UIDS = OrderedDict(
    V=['0', '1', '2', '3'],
    spike_state=['0', '1', '2']
)


def test_IO_init(mode):
    io = IO('test.h5', base_dir='./')
    assert io.path == Path.cwd().absolute() / 'test.h5'
    assert io.file_handle is None
    assert io.status == 'init'
    assert io.filename == 'test.h5'
    assert io.base_dir == Path.cwd().absolute()


def test_IO_open(mode):
    io = IO('test.h5', base_dir='./')
    assert not io.isopen
    f_handle = io.open(mode=mode)
    assert isinstance(f_handle, h5py.File)
    assert f_handle.mode in ['r', 'r+']
    assert io.isopen
    f_handle = io.open(mode=mode)
    assert isinstance(f_handle, h5py.File)
    assert f_handle.mode in ['r', 'r+']
    assert io.isopen
    io.close()
    assert not io.isopen

def test_IO_close(mode):
    io = IO('test.h5', base_dir='./')
    assert not io.isopen
    f_handle = io.open(mode=mode)
    assert isinstance(f_handle, h5py.File)
    assert io.isopen
    io.close()
    assert not io.isopen
    assert 'Closed' in io.file_handle.__repr__()
    io.close()
    assert not io.isopen
    assert 'Closed' in io.file_handle.__repr__()
    io.close()


def test_Input_init():
    inp = Input('test_in.h5', uids=INPUT_UIDS, data=INPUT_DATA, dt=dt, base_dir='./')
    assert inp.vars == ['I', 'g']
    actual_uids = dict(I=['0', '1', '2', '3'], g=['0', '1', '2'])
    for var in ['I', 'g']:
        assert all([a == b for a, b in zip(inp.uids[var], actual_uids[var])])
    assert inp.dt == dt
    with h5py.File(inp.path, 'r') as f:
        print(f['I/data'][()])
        np.testing.assert_equal(f['I/data'][()], INPUT_DATA['I'])
        np.testing.assert_equal(f['g/data'][()], INPUT_DATA['g'])
        np.testing.assert_equal(f['I/uids'][()].astype(str), np.array(INPUT_UIDS['I']).astype(str))
        np.testing.assert_equal(f['g/uids'][()].astype(str), np.array(INPUT_UIDS['g']).astype(str))
    os.remove(inp.path)


def test_Input_read():
    inp = Input('test_in.h5', uids=INPUT_UIDS, data=INPUT_DATA, dt=dt, base_dir='./')
    data = inp.read('0')
    assert 'I' in data and 'g' in data
    np.testing.assert_equal(data['I']['0'], INPUT_DATA['I'][:, [0]])

    data = inp.read('3')
    assert 'I' in data and 'g' not in data
    np.testing.assert_equal(data['I']['3'], INPUT_DATA['I'][:, [-1]])

    data = inp.read(['0', '3'])
    assert 'I' in data and 'g' in data
    np.testing.assert_equal(data['I']['0'], INPUT_DATA['I'][:, [0]])
    np.testing.assert_equal(data['I']['3'], INPUT_DATA['I'][:, [-1]])
    np.testing.assert_equal(data['g']['0'], INPUT_DATA['g'][:, [0]])
    os.remove(inp.path)

def test_Output_init():
    with h5py.File('test_out.h5', 'w') as f:
        f.create_dataset('V/uids', data=np.array(OUTPUT_UIDS['V']).astype('S'))
        f.create_dataset('V/data',
                         shape=OUTPUT_DATA['V'].shape,
                         dtype=OUTPUT_DATA['V'].dtype,
                         data=OUTPUT_DATA['V'])
        f.create_dataset('spike_state/uids', data=np.array(OUTPUT_UIDS['spike_state']).astype('S'))
        f.create_dataset('spike_state/data',
                         shape=OUTPUT_DATA['spike_state'].shape,
                         dtype=OUTPUT_DATA['spike_state'].dtype,
                         data=OUTPUT_DATA['spike_state'])
        f.create_dataset('metadata', (), 'i')
        f['metadata'].attrs['dt'] = dt

    out = Output(filename="test_out.h5",
                 uids={'spike_state':None, 'V':None})

    # before run complete
    assert out.status == 'init'
    assert not out.isReady
    assert all([k in out.uids for k in ['spike_state', 'V']])
    assert all([k in out.vars for k in ['spike_state', 'V']])
    assert out.dt is None
    assert out.t is None
    assert set(out.vars) == set(['spike_state', 'V'])

    # after pre_run
    out.status = 'pre_run'
    assert not out.isReady
    assert all([k in out.uids for k in ['spike_state', 'V']])
    assert all([k in out.vars for k in ['spike_state', 'V']])
    assert out.dt is None
    assert out.t is None
    assert set(out.vars) == set(['spike_state', 'V'])

    # after run
    out.status = 'run'
    assert out.isReady
    assert all([k in out.uids for k in ['spike_state', 'V']])
    assert out.dt == dt
    np.testing.assert_equal(out.t['V'], np.arange(1000)*dt)
    np.testing.assert_equal(out.t['spike_state'], np.arange(1000)*dt)
    assert set(out.vars) == set(['V', 'spike_state'])

    out.close()
    os.remove(out.path)

def test_Output_read():
    with h5py.File('test_out.h5', 'w') as f:
        f.create_dataset('V/uids', data=np.array(OUTPUT_UIDS['V']).astype('S'))
        f.create_dataset('V/data',
                         shape=OUTPUT_DATA['V'].shape,
                         dtype=OUTPUT_DATA['V'].dtype,
                         data=OUTPUT_DATA['V'])
        f.create_dataset('spike_state/uids', data=np.array(OUTPUT_UIDS['spike_state']).astype('S'))
        f.create_dataset('spike_state/data',
                         shape=OUTPUT_DATA['spike_state'].shape,
                         dtype=OUTPUT_DATA['spike_state'].dtype,
                         data=OUTPUT_DATA['spike_state'])
        f.create_dataset('metadata', (), 'i')
        f['metadata'].attrs['dt'] = dt

    out = Output(filename="test_out.h5",
                 uids={'spike_state':None, 'V':None})

    out.status = 'run'
    assert out.isReady

    data = out.read('0')
    assert 'V' in data and 'spike_state' in data
    np.testing.assert_equal(data['V']['0'], OUTPUT_DATA['V'][:, [0]])
    np.testing.assert_equal(data['spike_state']['0'], OUTPUT_DATA['spike_state'][:, [0]])

    data = out.read('0', vars=['V'])
    assert 'V' in data and 'spike_state' not in data
    np.testing.assert_equal(data['V']['0'], OUTPUT_DATA['V'][:, [0]])

    data = out.read('3')
    assert 'V' in data and 'spike_state' not in data
    np.testing.assert_equal(data['V']['3'], OUTPUT_DATA['V'][:, [-1]])

    data = out.read('3', vars='spike_state')
    assert not data # data should be empty

    data = out.read(['0', '3'])
    assert 'V' in data and 'spike_state' in data
    np.testing.assert_equal(data['V']['0'], OUTPUT_DATA['V'][:, [0]])
    np.testing.assert_equal(data['V']['3'], OUTPUT_DATA['V'][:, [-1]])
    np.testing.assert_equal(data['spike_state']['0'], OUTPUT_DATA['spike_state'][:, [0]])
    out.close()
    os.remove(out.path)
