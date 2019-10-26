import Uppy from '@uppy/core';
import Dashboard from '@uppy/dashboard';
import Webcam from '@uppy/Webcam';
import XHRUpload from '@uppy/xhr-upload';

export const uppy = Uppy({
  autoProceed: true,
  restrictions: {
    allowedFileTypes: ['image/*']
  }
});

uppy.use(Dashboard, {
  inline: true,
  target: '#drag-drop-area',
  replaceTargetContent: true
});
uppy.use(Webcam, {
  modes: ['picture'],
  facingMode: 'environment',
  target: Dashboard
});
uppy.use(XHRUpload, {endpoint: `http://${location.host}/upload-samples`});
