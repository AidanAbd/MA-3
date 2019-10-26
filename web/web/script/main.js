import * as ws from './ws';
import {uppy} from './uppy';

ws.useUppy(uppy);

document.addEventListener('DOMContentLoaded', () => {

});

const classNameField = document.getElementById('class-name-in');
const nextBtn = document.getElementById('next-btn');
nextBtn.addEventListener('click', () => {
  if (classNameField.value === '') {
    $('#no-class-name-modal').modal();
    return;
  }

  ws.setWorkingsetClassName(classNameField.value);
});
