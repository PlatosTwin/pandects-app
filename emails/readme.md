# All things email

We use [Resend](https://resend.com/) and [React Email](https://react.email/) to create email templates—e.g., a React Email template sits behind the "Verify your email" message new non-Google users get upon creating an account.

## Getting Started

First, install the dependencies:

```sh
npm install
# or
yarn
```

Then, run the development server:

```sh
npm run dev
# or
yarn dev
```

Open [localhost:3000](http://localhost:3000) with your browser to see available templates.

## Uploading templates to Resend

To upload a template to Resend:
```
npm run email -- --template-id <template id>
```

If the template ID doesn't exist, the script will create a new template. If it exists but isn't published, the script will throw an error. If it exists and is published, the script will update the template, putting the updates into draft mode.

## License

MIT License
