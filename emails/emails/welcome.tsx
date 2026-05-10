import {
  Body,
  Container,
  Head,
  Hr,
  Html,
  Img,
  Link,
  Preview,
  Section,
  Text,
} from '@react-email/components';

export interface WelcomeProps {
  NAME: string;
  YEAR: string;
}

export const Welcome = ({ NAME, YEAR }: WelcomeProps) => {
  const previewText = 'Thanks for signing up for Pandects';

  return (
    <Html>
      <Head />
      <Preview>{previewText}</Preview>
      <Body style={main}>
        <Container style={container}>
          <Section style={brandSection}>
            <Text style={brandText}>Pandects</Text>
          </Section>

          <Section style={logoSection}>
            <Img
              src="https://raw.githubusercontent.com/PlatosTwin/pandects-app/main/frontend/assets/logo-256.png"
              width="96"
              height="96"
              alt="Pandects panda"
              style={logoImg}
            />
          </Section>

          <Text style={copy}>Hi {NAME},</Text>

          <Text style={copy}>
            Thanks for signing up for an account with Pandects!
          </Text>

          <Text style={copy}>
            As you browse the platform, keep in mind that this is an{' '}
            <Link
              href="https://github.com/PlatosTwin/pandects-app"
              style={underlineLink}
            >
              open-source project
            </Link>
            , meaning that if you&apos;ve spotted a bug or would like to contribute a feature, you
            can open an{' '}
            <Link
              href="https://github.com/PlatosTwin/pandects-app/issues"
              style={inlineLink}
            >
              issue
            </Link>{' '}
            or{' '}
            <Link
              href="https://github.com/PlatosTwin/pandects-app/pulls"
              style={inlineLink}
            >
              PR
            </Link>{' '}
            on GitHub&mdash;or, feel free to respond here, which is my personal email address.
          </Text>

          <Text style={signoff}>
            All best,
            <br />
            <br />
            Nikita
            <br />
            Keeper of the Agreements
          </Text>

          <Hr style={divider} />

          <Text style={footer}>© {YEAR} Pandects</Text>
        </Container>
      </Body>
    </Html>
  );
};

Welcome.PreviewProps = {
  NAME: 'Alex',
  YEAR: '{{{YEAR}}}',
} as WelcomeProps;

export default Welcome;

const main = {
  backgroundColor: '#eef1f6',
  fontFamily:
    'ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif',
  padding: '12px',
};

const container = {
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0',
  margin: '44px auto',
  maxWidth: '520px',
  padding: '32px',
};

const brandSection = {
  marginBottom: '20px',
};

const brandText = {
  color: '#0f172a',
  fontSize: '14px',
  fontWeight: 600,
  margin: '0',
};

const logoSection = {
  margin: '20px 0 24px',
  textAlign: 'center' as const,
};

const logoImg = {
  border: '1px solid #d1d9e6',
  borderRadius: '12px',
  display: 'block',
  margin: '0 auto',
};

const copy = {
  color: '#344964',
  fontSize: '14px',
  lineHeight: '22px',
  margin: '0 0 16px',
};

const underlineLink = {
  color: '#344964',
  textDecoration: 'underline',
};

const inlineLink = {
  color: '#2563eb',
  textDecoration: 'none',
};

const signoff = {
  color: '#344964',
  fontSize: '14px',
  lineHeight: '22px',
  margin: '0 0 16px',
};

const divider = {
  borderColor: '#e2e8f0',
  margin: '24px 0',
};

const footer = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  margin: '0',
};
