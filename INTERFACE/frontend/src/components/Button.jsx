function Button({ children, onClick, disabled, className, variant = "default", size = "md" }) {
    const baseClasses = "font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2"
  
    const variantClasses = {
      default: "bg-gray-200 hover:bg-gray-300 text-gray-800 focus:ring-gray-400",
      primary: "bg-emerald-500 hover:bg-emerald-600 text-white focus:ring-emerald-400",
      secondary: "bg-sky-500 hover:bg-sky-600 text-white focus:ring-sky-400",
      outline: "border border-gray-300 hover:bg-gray-100 text-gray-800 focus:ring-gray-400",
      destructive: "bg-red-500 hover:bg-red-600 text-white focus:ring-red-400",
      emerald: "bg-emerald-300 hover:bg-emerald-400 text-black focus:ring-emerald-400",
      sky: "bg-sky-400 hover:bg-sky-500 text-black focus:ring-sky-400",
    }
  
    const sizeClasses = {
      sm: "px-3 py-1 text-sm",
      md: "px-4 py-2",
      lg: "px-6 py-3 text-lg",
    }
  
    const classes = `
      ${baseClasses} 
      ${variantClasses[variant]} 
      ${sizeClasses[size]} 
      ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"} 
      ${className || ""}
    `
  
    return (
      <button className={classes} onClick={onClick} disabled={disabled}>
        {children}
      </button>
    )
  }
  
  export default Button
  