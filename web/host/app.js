import fs from 'fs-extra';
import path from 'path';
import util from 'util';
import child_process from 'child_process';

import Koa from 'koa';
import koaBody from 'koa-body';
import KoaLogger from 'koa-logger';
import compress from 'koa-compress';
import send from 'koa-send';
import websocket from 'koa-easy-ws';

import {handleWs} from './ws';
import {getSession, newSession} from './user-session';

const execFile = util.promisify(child_process.execFile);


const distPath = fs.realpathSync(path.join(__dirname, '../web/dist'));

const app = new Koa();

app.use(new KoaLogger());
app.use(websocket());
app.use(compress());

const koaBodyMW = koaBody({multipart: true});

app.use(async (ctx, next)=>{
  if (ctx.ws != null) {
    const ws = await ctx.ws();

    const sess = newSession();
    handleWs(ws, sess);

    return;
  }

  if (ctx.path === '/') {
    await send(ctx, 'index.html', {root: distPath});
    return;
  }

  if (ctx.path === '/upload-samples') {
    await koaBodyMW.call(this, ctx, next);

    const req = ctx.request;

    console.log(req.body);

    if (req.body.payload == null) {
      console.error('No payload');
      ctx.status = 400;
      return;
    }
    if (req.body.label == null) {
      console.error('No label');
      ctx.status = 400;
      return;
    }
    if (req.files == null) {
      console.error('No files');
      ctx.status = 400;
      return;
    }

    const sess = await getSession(req.body.payload);

    if (sess == null) {
      ctx.status = 403;
      return;
    }

    console.log(req.files['files[]']);
    // await fs.promise.rename();

    ctx.status = 200;

    return;
  }

  try {
    await send(ctx, ctx.path, {root: distPath});
  }
  catch (e) {
    ctx.status = 404;
  }
});

(async () => {
  await execFile('./rsa-gen.sh');

  app.listen(3000);
  console.log('listening');
})();
