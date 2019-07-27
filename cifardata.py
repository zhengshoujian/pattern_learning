class CifarData:
    def __init__( self, filenames, need_shuffle ):
        all_data = []
        all_labels = []

        for filename in filenames:
            data, labels = load_data( filename )
            all_data.append( data )
            all_labels.append( labels )

        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack( all_labels )
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shffle_data()

    def _shffle_data( self ):
        p = np.random.permutation( self._num_examples )
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch( self, batch_size ):
        '''return batch_size example as a batch'''
        end_indictor = self._indicator + batch_size
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                self._shffle_data()
                self._indicator = 0
                end_indictor = batch_size # 其实就是 0 + batch_size, 把 0 省略了
            else:
                raise Exception( "have no more examples" )
        if end_indictor > self._num_examples:
            raise Exception( "batch size is larger than all example" )

        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels