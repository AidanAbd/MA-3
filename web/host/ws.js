export const handleWs = (ws, sess) => {
  ws.on('close', async () => {
    await sess.cleanup();
  });

  let packetCounter = 0;

  const send = (type, data)=>{
    ws.send(JSON.stringify({
      type: type,
      id: packetCounter++,
      data
    }));
  };
  const sendAck = (obj)=>{
    send('ack', {id: obj.id});
  };

  const errorMessages = {
    'bad-packet-json': 'Received packet that is invalid JSON.',
    'null-packet': 'Received a null packet.',
    'no-such-packet': 'Received packet with invalid type: "${packet}".',
    'no-such-id': 'Received a ${packet_type} for invalid packet with id ${id}.',
    'missing-protocol-field': 'Received a ${packet_type} without the required protocol field ${field_name}.',
    'missing-body-field': 'Received a ${packet_type} without the required body field ${field_name}.',
    'field-out-of-range': 'Field ${field_name} is unexpectedly not in range ${range_string}.',
    'field-of-wrong-type': 'Field ${field_name} is of invalid type, expected a ${expected_type}.',
    'bug': 'Something went wrong on the receiving side. This is a bug.',
    'wrong-pong': 'Received a pong with the wrong message. Expected: "${message}", not "${wrong_message}".',
    'path-does-not-exist': '${path} unexpectedly does not exist.',
    'unexpected-empty-string': 'Field ${field_name} is unexpectedly empty.',
    'unexpected-class': '"${class}" is not one of previously specified classes.'
  };
  const sendError = (cause, code, etc)=>{
    send('error', {
      id: cause,
      code: code,
      etc: etc == null ? '' : etc
    });
  };

  ws.on('message', async (data) => {
    let obj = null;
    try {
      obj = JSON.parse(data);
    }
    catch (e) {
      sendError(obj.id, 'bad-json');
      return;
    }
    if (obj == null) {
      sendError(obj.id, 'null-packet');
      return;
    }

    if (obj.type == null) {
      sendError(obj.id, 'missing-protocol-field', 'type');
      return;
    }
    if (obj.id == null) {
      sendError(obj.id, 'missing-protocol-field', 'id');
      return;
    }

    const requireBodyField = (packet, field) => {
      if (packet.data == null) {
        sendError(packet.id, 'missing-protocol-field', 'data');
        return false;
      }

      if (packet.data[field] == null) {
        sendError(packet.id, 'missing-body-field', field);
        return false;
      }

      return true;
    };

    if (obj.type === 'ack') {
      return;
    }
    else if (obj.type === 'error') {
      if (!requireBodyField(obj, 'id')) return;
      if (!requireBodyField(obj, 'code')) return;
      if (!requireBodyField(obj, 'etc')) return;

      let msg = `Packet ${obj.data.id} caused ${obj.data.code}`;

      if (obj.etc !== '')
        msg += ` "${obj.data.etc}"`;

      console.error(msg);

      sendAck(obj);

      return;
    }
    else if (obj.type === 'ping') {
      if (!requireBodyField(obj, 'message')) return;

      send('pong', {id: obj.id, message: obj.data.message});
      return;
    }
    else if (obj.type === 'pong') {
      if (!requireBodyField(obj, 'message')) return;
      console.log(`Pong! ${obj.data.message}`);

      sendAck(obj);

      return;
    }
    else if (obj.type === 'progress') {
      if (!requireBodyField(obj, 'completeness')) return;

      console.log(`${obj.data.id}: ${obj.data.completeness*100}%`);

      sendAck(obj);

      return;
    }
    else if (obj.type === 'auth') {
      const payload = await sess.jwtpayload();
      send('auth-response', {payload});

      console.log('Auth');

      return;
    }
    else if (obj.type === 'working-set-class') {
      if (!requireBodyField(obj, 'label')) return;

      await sess.setWorkingsetClassName(obj.data.label);

      sendAck(obj);
      return;
    }
    else if (obj.type === 'start-training') {
      await sess.startTraining();

      sendAck(obj);
      return;
    }
    else if (obj.type === 'remove-class') {
      if (!requireBodyField(obj, 'label')) return;

      await sess.removeClass(obj.data.label);

      sendAck(obj);
      return;
    }

    sendError(obj.id, 'no-such-packet');
  });

  sess.setWS({
    sendProgress: (completeness) => {
      send('progress', {completeness});
    }
  });
};
