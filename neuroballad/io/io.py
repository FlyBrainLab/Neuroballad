'''Parent Class of IOs related to NeuroDriver

This class handles io for reading/writing input/output h5 files 
for simulating a NeuroDriver session.
'''
import logging
from abc import abstractmethod
from pathlib import Path
import atexit
import h5py

class IO(object):
    def __init__(self, filename, base_dir='./'):
        '''
        Parameters
        ----------
        filename: string
            file name of IO
        base_dir: string, optional
            the base directory that holds all io files
        '''
        self.filename = filename
        self.status = 'init' # keeps track of whether file loaded with data yet['init', 'pre_run', 'run']
        self.base_dir = Path(base_dir).absolute()
        self.file_handle = None

        @atexit.register
        def close_everything():
            if isinstance(self.file_handle, h5py.File):
                if self.isopen:
                    self.file_handle.flush()
                self.file_handle.close()

        try:
            self.open(mode='w')
            self.close()
        except Exception as e:
            self.close()
            logging.exception('Test file opening failed, aborting: {}'.format(e))

    @property
    def path(self):
        '''Return Path to file'''
        return self.base_dir / self.filename

    @property
    def isopen(self):
        '''Check if file is open

        returns `True` if is open
        '''
        if self.file_handle is not None:
            if isinstance(self.file_handle, h5py.File):
                if 'Closed' in self.file_handle.__repr__(): # monkey hack to check if file is closed
                    return False
                return True
            else:
                raise TypeError('file_handle of type {} not understood'.format(type(self.file_handle)))
        return False

    def open(self, mode='a'):
        '''open file for reading
        
        if not open then open

        Returns
        -------
        file_handle: h5py._hl.files.File
            handle to h5py file object
        '''
        while not self.isopen:
            self.file_handle = h5py.File(self.path, mode=mode)
        return self.file_handle

    def close(self):
        '''Close file 
        
        if not closed then open

        Returns
        -------
        file_handle: h5py._hl.files.File
            handle to h5py file object
        '''
        if self.file_handle is not None:
            if self.isopen:
                self.file_handle.flush()
            self.file_handle.close()
        return self.file_handle

    @abstractmethod
    def read(self, nodes_ids, vars=None, prune_empty=True):
        '''Plot All information related to given node'''

    @abstractmethod
    def plot(self, nodes_ids, **kwargs):
        '''Plot All information related to given node'''
