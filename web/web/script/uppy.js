import Uppy from '@uppy/core';
import Dashboard from '@uppy/dashboard';
import Webcam from '@uppy/webcam';
import XHRUpload from '@uppy/xhr-upload';
import ThumbnailGenerator from '@uppy/thumbnail-generator';

export const setupUppyFor = (dragAndDropAreaId, apiEndpoint) => {
  const uppy = Uppy({
    autoProceed: true,
    restrictions: {
      allowedFileTypes: ['image/*']
    }
  });

  uppy.use(Dashboard, {
    inline: true,
    target: `#${dragAndDropAreaId}`,
    replaceTargetContent: true
  });
  uppy.use(Webcam, {
    modes: ['picture'],
    facingMode: 'environment',
    target: Dashboard
  });
  uppy.use(ThumbnailGenerator, {
    waitForThumbnailsBeforeUpload: true
  });
  uppy.use(XHRUpload, {endpoint: `http://${location.host}/${apiEndpoint}`});

  return uppy;
};
