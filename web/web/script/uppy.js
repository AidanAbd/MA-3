import Uppy from '@uppy/core';
import Dashboard from '@uppy/dashboard';
import XHRUpload from '@uppy/xhr-upload';

export const uppy = Uppy({
  autoProceed: true
});

uppy.use(Dashboard, {
  inline: true,
  target: '#drag-drop-area'
});
uppy.use(XHRUpload, {endpoint: `http://${location.host}/upload-samples`});
