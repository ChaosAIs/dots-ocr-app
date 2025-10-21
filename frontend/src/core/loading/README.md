# Loading Service Control

The loading service now supports conditional enabling/disabling of the loading interceptor to prevent unwanted loading spinners in specific contexts.

## Features

- **Global Control**: Enable/disable loading interceptor globally
- **Context-based Control**: Disable loading for specific contexts (e.g., 'survey', 'form')
- **React Hooks**: Easy-to-use hooks for component-level control
- **Automatic Cleanup**: Proper cleanup when components unmount

## Usage

### 1. Using React Hooks (Recommended)

#### Disable loading for a specific context:
```javascript
import { useLoadingControl } from '../core/loading/hooks/useLoadingControl';

const MyComponent = () => {
  // Disable loading interceptor for 'survey' context
  useLoadingControl('survey', true);
  
  // Your component logic...
  return <div>My Component</div>;
};
```

#### Global loading control:
```javascript
import { useGlobalLoadingControl } from '../core/loading/hooks/useLoadingControl';

const MyComponent = () => {
  // Disable loading interceptor globally
  const loadingControl = useGlobalLoadingControl(false);
  
  // Manual control
  const handleDisable = () => loadingControl.disable();
  const handleEnable = () => loadingControl.enable();
  
  return (
    <div>
      <button onClick={handleDisable}>Disable Loading</button>
      <button onClick={handleEnable}>Enable Loading</button>
    </div>
  );
};
```

### 2. Direct Service Usage

#### Context-based control:
```javascript
import { loadingService } from '../core/loading/loadingService';

// Disable loading for 'survey' context
loadingService.disableForContext('survey');

// Re-enable loading for 'survey' context
loadingService.enableForContext('survey');

// Check if context is disabled
const isDisabled = loadingService.isContextDisabled('survey');
```

#### Global control:
```javascript
import { loadingService } from '../core/loading/loadingService';

// Disable loading interceptor globally
loadingService.setInterceptorEnabled(false);

// Enable loading interceptor globally
loadingService.setInterceptorEnabled(true);

// Check if interceptor is enabled
const isEnabled = loadingService.isInterceptorEnabled();
```

## API Reference

### loadingService Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `setInterceptorEnabled(enabled)` | Enable/disable interceptor globally | `boolean` | `void` |
| `isInterceptorEnabled()` | Check if interceptor is enabled | none | `boolean` |
| `disableForContext(context)` | Disable for specific context | `string` | `void` |
| `enableForContext(context)` | Enable for specific context | `string` | `void` |
| `isContextDisabled(context)` | Check if context is disabled | `string` | `boolean` |
| `clearDisabledContexts()` | Clear all disabled contexts | none | `void` |
| `getDisabledContexts()` | Get disabled contexts | none | `Set<string>` |

### useLoadingControl Hook

```javascript
const loadingControl = useLoadingControl(context, disabled);
```

**Parameters:**
- `context` (string): Context identifier
- `disabled` (boolean, default: true): Whether to disable loading

**Returns:** Object with control methods

### useGlobalLoadingControl Hook

```javascript
const globalControl = useGlobalLoadingControl(enabled);
```

**Parameters:**
- `enabled` (boolean, default: true): Whether loading should be enabled

**Returns:** Object with global control methods

## Examples

### Survey Component (Current Implementation)
```javascript
import { useLoadingControl } from '../core/loading/hooks/useLoadingControl';

export const SurveyHome = () => {
  // Disable loading interceptor for survey component
  useLoadingControl('survey', true);
  
  // Component logic...
};
```

### Form Component
```javascript
import { useLoadingControl } from '../core/loading/hooks/useLoadingControl';

export const FormComponent = () => {
  // Disable loading for form context
  useLoadingControl('form', true);
  
  // Component logic...
};
```

### Conditional Loading Control
```javascript
import { useLoadingControl } from '../core/loading/hooks/useLoadingControl';

export const ConditionalComponent = () => {
  const [shouldDisableLoading, setShouldDisableLoading] = useState(false);
  
  // Conditionally disable loading
  useLoadingControl('conditional', shouldDisableLoading);
  
  return (
    <div>
      <button onClick={() => setShouldDisableLoading(!shouldDisableLoading)}>
        {shouldDisableLoading ? 'Enable' : 'Disable'} Loading
      </button>
    </div>
  );
};
```

## How It Works

1. **HTTP Interceptor Check**: Before showing/hiding loading spinner, the service checks if interceptor should be applied
2. **Global Flag**: If globally disabled, no loading spinner is shown
3. **Context Check**: If any disabled context is active, no loading spinner is shown
4. **Automatic Cleanup**: React hooks automatically clean up when components unmount

## Best Practices

1. Use React hooks for component-level control
2. Use descriptive context names (e.g., 'survey', 'form', 'upload')
3. Always clean up contexts when components unmount (hooks do this automatically)
4. Prefer context-based control over global control for better granularity
5. Test loading behavior in different scenarios to ensure proper UX
